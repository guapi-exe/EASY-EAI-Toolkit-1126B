#include "streaming/rtsp_streamer.h"

#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

extern "C" {
#include "log.h"
}

namespace {

bool ensureGstInitialized() {
    static std::once_flag initFlag;
    static bool initialized = false;

    std::call_once(initFlag, []() {
        GError* error = nullptr;
        initialized = gst_init_check(nullptr, nullptr, &error);
        if (!initialized) {
            log_error("RTSP: gst_init_check failed: %s",
                      error ? error->message : "unknown error");
        }
        if (error) {
            g_error_free(error);
        }
    });

    return initialized;
}

bool hasGstFactory(const char* name) {
    GstElementFactory* factory = gst_element_factory_find(name);
    if (!factory) {
        return false;
    }
    gst_object_unref(factory);
    return true;
}

bool elementHasProperty(const char* factoryName, const char* propertyName) {
    GstElement* element = gst_element_factory_make(factoryName, nullptr);
    if (!element) {
        return false;
    }
    bool hasProperty =
        g_object_class_find_property(G_OBJECT_GET_CLASS(element), propertyName) != nullptr;
    gst_object_unref(element);
    return hasProperty;
}

std::string buildH264EncoderLaunch(const RtspStreamerConfig& config,
                                   std::string* encoderName) {
    const int gop = std::max(1, config.fps * 2);
    const std::string h264parse = hasGstFactory("h264parse") ? " ! h264parse" : "";

    if (hasGstFactory("mpph264enc")) {
        std::string encoder = "mpph264enc";
        if (elementHasProperty("mpph264enc", "bps")) {
            encoder += " bps=" + std::to_string(std::max(1, config.bitrateKbps) * 1000);
        } else if (elementHasProperty("mpph264enc", "bitrate")) {
            encoder += " bitrate=" + std::to_string(std::max(1, config.bitrateKbps));
        }
        if (elementHasProperty("mpph264enc", "gop")) {
            encoder += " gop=" + std::to_string(gop);
        }
        if (encoderName) {
            *encoderName = "mpph264enc";
        }
        return "videoconvert ! video/x-raw,format=NV12 ! " +
               encoder +
               h264parse + " ! ";
    }

    if (hasGstFactory("v4l2h264enc")) {
        std::string encoder = "v4l2h264enc";
        if (elementHasProperty("v4l2h264enc", "bitrate")) {
            encoder += " bitrate=" + std::to_string(std::max(1, config.bitrateKbps) * 1000);
        }
        if (encoderName) {
            *encoderName = "v4l2h264enc";
        }
        return "videoconvert ! video/x-raw,format=NV12 ! " +
               encoder +
               h264parse + " ! ";
    }

    if (encoderName) {
        *encoderName = "x264enc";
    }
    char launch[256];
    std::snprintf(launch,
                  sizeof(launch),
                  "videoconvert ! x264enc tune=zerolatency bitrate=%d "
                  "speed-preset=ultrafast key-int-max=%d bframes=0 byte-stream=true ! ",
                  std::max(1, config.bitrateKbps),
                  gop);
    return std::string(launch);
}

void configureAppSrc(GstElement* appsrc,
                     const RtspStreamerConfig& config,
                     gsize frameSize) {
    g_object_set(G_OBJECT(appsrc),
                 "is-live", TRUE,
                 "do-timestamp", TRUE,
                 "format", GST_FORMAT_TIME,
                 "block", FALSE,
                 "max-bytes", frameSize * 2,
                 NULL);

    GObjectClass* klass = G_OBJECT_GET_CLASS(appsrc);
    if (g_object_class_find_property(klass, "max-buffers")) {
        g_object_set(G_OBJECT(appsrc), "max-buffers", 2, NULL);
    }
    if (g_object_class_find_property(klass, "leaky-type")) {
        g_object_set(G_OBJECT(appsrc), "leaky-type", 2, NULL);
    }

    GstCaps* caps = gst_caps_new_simple("video/x-raw",
                                        "format", G_TYPE_STRING, "BGR",
                                        "width", G_TYPE_INT, config.width,
                                        "height", G_TYPE_INT, config.height,
                                        "framerate", GST_TYPE_FRACTION, config.fps, 1,
                                        NULL);
    g_object_set(G_OBJECT(appsrc), "caps", caps, NULL);
    gst_caps_unref(caps);
}

} // namespace

struct RtspStreamer::Impl {
    explicit Impl(const RtspStreamerConfig& cfg) : config(cfg) {}

    RtspStreamerConfig config;
    GstRTSPServer* server{nullptr};
    GstRTSPMediaFactory* factory{nullptr};
    bool factoryOwnedByMount{false};
    GMainContext* context{nullptr};
    GMainLoop* loop{nullptr};
    guint sourceId{0};

    std::thread loopThread;
    std::atomic<bool> running{false};
    std::atomic<unsigned long long> droppedFrames{0};

    mutable std::mutex appsrcMutex;
    GstElement* appsrc{nullptr};
    unsigned long long frameCount{0};

    static void mediaUnprepared(GstRTSPMedia* media, gpointer userData) {
        (void)media;
        Impl* self = static_cast<Impl*>(userData);
        if (!self) {
            return;
        }

        std::lock_guard<std::mutex> lock(self->appsrcMutex);
        if (self->appsrc) {
            gst_object_unref(self->appsrc);
            self->appsrc = nullptr;
            log_info("RTSP: client disconnected");
        }
    }

    static void mediaConfigure(GstRTSPMediaFactory* factory,
                               GstRTSPMedia* media,
                               gpointer userData) {
        (void)factory;
        Impl* self = static_cast<Impl*>(userData);
        if (!self || !self->running) {
            return;
        }

        GstElement* element = gst_rtsp_media_get_element(media);
        if (!element) {
            log_error("RTSP: failed to get media element");
            return;
        }

        GstElement* videosrc = gst_bin_get_by_name(GST_BIN(element), "videosrc");
        if (!videosrc) {
            log_error("RTSP: failed to find appsrc 'videosrc'");
            gst_object_unref(element);
            return;
        }

        gsize frameSize = static_cast<gsize>(self->config.width) *
                          static_cast<gsize>(self->config.height) * 3;
        configureAppSrc(videosrc, self->config, frameSize);
        g_signal_connect(media, "unprepared", G_CALLBACK(mediaUnprepared), self);

        {
            std::lock_guard<std::mutex> lock(self->appsrcMutex);
            if (self->appsrc) {
                gst_object_unref(self->appsrc);
            }
            self->appsrc = GST_ELEMENT(gst_object_ref(videosrc));
            self->frameCount = 0;
        }

        log_info("RTSP: client connected, stream=%dx%d@%dfps",
                 self->config.width,
                 self->config.height,
                 self->config.fps);

        gst_object_unref(videosrc);
        gst_object_unref(element);
    }

    bool start() {
        if (running) {
            return true;
        }
        if (!ensureGstInitialized()) {
            return false;
        }

        context = g_main_context_new();
        loop = g_main_loop_new(context, FALSE);
        server = gst_rtsp_server_new();
        if (!context || !loop || !server) {
            log_error("RTSP: failed to allocate server context");
            stop();
            return false;
        }

        g_object_set(server, "service", config.port.c_str(), NULL);

        GstRTSPMountPoints* mounts = gst_rtsp_server_get_mount_points(server);
        factory = gst_rtsp_media_factory_new();
        if (!mounts || !factory) {
            log_error("RTSP: failed to create media factory");
            if (mounts) {
                g_object_unref(mounts);
            }
            stop();
            return false;
        }

        std::string encoderName;
        std::string encoderLaunch = buildH264EncoderLaunch(config, &encoderName);
        char launch[1024];
        std::snprintf(launch,
                      sizeof(launch),
                      "appsrc name=videosrc format=time is-live=true do-timestamp=true block=false ! "
                      "queue max-size-buffers=1 leaky=downstream ! "
                      "%s"
                      "rtph264pay name=pay0 pt=96 config-interval=1",
                      encoderLaunch.c_str());
        log_info("RTSP: encoder=%s, stream=%dx%d@%dfps bitrate=%dkbps",
                 encoderName.c_str(),
                 config.width,
                 config.height,
                 config.fps,
                 config.bitrateKbps);

        gst_rtsp_media_factory_set_launch(factory, launch);
        gst_rtsp_media_factory_set_shared(factory, TRUE);
        g_signal_connect(factory, "media-configure", G_CALLBACK(mediaConfigure), this);

        std::string mountPath = config.mountPath.empty() ? "/stream" : config.mountPath;
        if (mountPath[0] != '/') {
            mountPath.insert(mountPath.begin(), '/');
        }
        gst_rtsp_mount_points_add_factory(mounts, mountPath.c_str(), factory);
        factoryOwnedByMount = true;
        g_object_unref(mounts);

        sourceId = gst_rtsp_server_attach(server, context);
        if (sourceId == 0) {
            log_error("RTSP: failed to attach server");
            stop();
            return false;
        }

        running = true;
        loopThread = std::thread([this]() {
            g_main_context_push_thread_default(context);
            g_main_loop_run(loop);
            g_main_context_pop_thread_default(context);
        });

        log_info("RTSP: server started at %s", url().c_str());
        return true;
    }

    void stop() {
        running = false;

        {
            std::lock_guard<std::mutex> lock(appsrcMutex);
            if (appsrc) {
                gst_app_src_end_of_stream(GST_APP_SRC(appsrc));
                gst_object_unref(appsrc);
                appsrc = nullptr;
            }
        }

        if (loop) {
            g_main_loop_quit(loop);
        }
        if (loopThread.joinable()) {
            loopThread.join();
        }

        if (sourceId != 0) {
            g_source_remove(sourceId);
            sourceId = 0;
        }
        if (factory && !factoryOwnedByMount) {
            g_object_unref(factory);
        }
        factory = nullptr;
        factoryOwnedByMount = false;
        if (server) {
            g_object_unref(server);
            server = nullptr;
        }
        if (loop) {
            g_main_loop_unref(loop);
            loop = nullptr;
        }
        if (context) {
            g_main_context_unref(context);
            context = nullptr;
        }
    }

    bool pushFrame(const cv::Mat& input) {
        if (!running || input.empty()) {
            return false;
        }

        GstElement* localAppsrc = nullptr;
        unsigned long long frameIndex = 0;
        {
            std::lock_guard<std::mutex> lock(appsrcMutex);
            if (!appsrc) {
                return true;
            }
            localAppsrc = GST_ELEMENT(gst_object_ref(appsrc));
            frameIndex = frameCount++;
        }

        cv::Mat resized;
        const cv::Mat* frame = &input;
        if (input.cols != config.width || input.rows != config.height) {
            cv::resize(input, resized, cv::Size(config.width, config.height), 0, 0, cv::INTER_LINEAR);
            frame = &resized;
        }
        if (!frame->isContinuous()) {
            resized = frame->clone();
            frame = &resized;
        }

        gsize size = static_cast<gsize>(frame->total() * frame->elemSize());
        GstBuffer* buffer = gst_buffer_new_allocate(nullptr, size, nullptr);
        if (!buffer) {
            gst_object_unref(localAppsrc);
            return false;
        }

        GstMapInfo map;
        if (!gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
            gst_buffer_unref(buffer);
            gst_object_unref(localAppsrc);
            return false;
        }
        std::memcpy(map.data, frame->data, size);
        gst_buffer_unmap(buffer, &map);

        GST_BUFFER_PTS(buffer) = gst_util_uint64_scale(frameIndex, GST_SECOND, config.fps);
        GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale(1, GST_SECOND, config.fps);

        GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(localAppsrc), buffer);
        gst_object_unref(localAppsrc);

        if (ret == GST_FLOW_OK) {
            return true;
        }
        if (ret != GST_FLOW_FLUSHING && ret != GST_FLOW_EOS) {
            unsigned long long dropped = ++droppedFrames;
            if (dropped % 100 == 1) {
                log_warn("RTSP: dropped frames=%llu, flow=%d", dropped, static_cast<int>(ret));
            }
        }
        return true;
    }

    bool isRunning() const {
        return running.load();
    }

    bool hasClient() const {
        std::lock_guard<std::mutex> lock(appsrcMutex);
        return appsrc != nullptr;
    }

    std::string url() const {
        std::string mountPath = config.mountPath.empty() ? "/stream" : config.mountPath;
        if (mountPath[0] != '/') {
            mountPath.insert(mountPath.begin(), '/');
        }
        return "rtsp://localhost:" + config.port + mountPath;
    }
};

RtspStreamer::RtspStreamer(const RtspStreamerConfig& config)
    : impl(new Impl(config)) {}

RtspStreamer::~RtspStreamer() {
    stop();
}

bool RtspStreamer::start() {
    return impl->start();
}

void RtspStreamer::stop() {
    if (impl) {
        impl->stop();
    }
}

bool RtspStreamer::pushFrame(const cv::Mat& frame) {
    return impl && impl->pushFrame(frame);
}

bool RtspStreamer::isRunning() const {
    return impl && impl->isRunning();
}

bool RtspStreamer::hasClient() const {
    return impl && impl->hasClient();
}

std::string RtspStreamer::url() const {
    return impl ? impl->url() : std::string();
}
