#include "streaming/rtsp_stream_task.h"

#include "streaming/stream_overlay.h"

#include <algorithm>
#include <chrono>
#include <utility>

extern "C" {
#include "log.h"
}

RtspStreamTask::RtspStreamTask(const RtspStreamOptions& opts)
    : options(opts) {
    streamerConfig.width = std::max(160, options.width);
    streamerConfig.height = std::max(120, options.height);
    streamerConfig.fps = std::max(1, options.fps);
    streamerConfig.bitrateKbps = std::max(256, options.bitrateKbps);
    streamerConfig.port = options.port.empty() ? "8554" : options.port;
    streamerConfig.mountPath = options.mountPath.empty() ? "/stream" : options.mountPath;
    if (streamerConfig.mountPath[0] != '/') {
        streamerConfig.mountPath.insert(streamerConfig.mountPath.begin(), '/');
    }
    minFrameIntervalMs.store(std::max<int64_t>(1, 1000 / streamerConfig.fps));
}

RtspStreamTask::~RtspStreamTask() {
    stop();
}

bool RtspStreamTask::start() {
    if (running) {
        return true;
    }

    streamer.reset(new RtspStreamer(streamerConfig));
    if (!streamer->start()) {
        streamer.reset();
        return false;
    }

    running = true;
    worker = std::thread(&RtspStreamTask::run, this);
    log_info("RtspStreamTask: started, url=%s", url().c_str());
    return true;
}

void RtspStreamTask::stop() {
    running = false;
    frameCv.notify_all();
    if (worker.joinable()) {
        worker.join();
    }
    if (streamer) {
        streamer->stop();
        streamer.reset();
    }
}

int64_t RtspStreamTask::nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

bool RtspStreamTask::shouldAcceptFrame() {
    if (!running) {
        return false;
    }
    if (!streamer || !streamer->hasClient()) {
        return false;
    }

    const int64_t now = nowMs();
    const int64_t interval = std::max<int64_t>(1, minFrameIntervalMs.load());
    int64_t last = lastAcceptedFrameMs.load();
    while (now - last >= interval) {
        if (lastAcceptedFrameMs.compare_exchange_weak(last, now)) {
            return true;
        }
    }
    return false;
}

void RtspStreamTask::publishFrame(const cv::Mat& frame,
                                  std::vector<StreamOverlayTrack> tracks,
                                  const StreamOverlayStats& stats) {
    if (!running || frame.empty()) {
        return;
    }

    std::unique_lock<std::mutex> lock(frameMutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        return;
    }

    latestFrame = frame.clone();
    latestTracks = std::move(tracks);
    latestStats = stats;
    latestSeq++;
    lock.unlock();
    frameCv.notify_one();
}

bool RtspStreamTask::isRunning() const {
    return running.load();
}

bool RtspStreamTask::hasClient() const {
    return streamer && streamer->hasClient();
}

std::string RtspStreamTask::url() const {
    return streamer ? streamer->url() :
        ("rtsp://localhost:" + streamerConfig.port + streamerConfig.mountPath);
}

void RtspStreamTask::run() {
    while (running) {
        cv::Mat frame;
        std::vector<StreamOverlayTrack> tracks;
        StreamOverlayStats stats;

        {
            std::unique_lock<std::mutex> lock(frameMutex);
            frameCv.wait_for(lock, std::chrono::milliseconds(250), [this]() {
                return !running || latestSeq != consumedSeq;
            });

            if (!running) {
                break;
            }
            if (latestSeq == consumedSeq || latestFrame.empty()) {
                continue;
            }

            frame = latestFrame;
            latestFrame.release();
            tracks = std::move(latestTracks);
            latestTracks.clear();
            stats = latestStats;
            consumedSeq = latestSeq;
        }

        drawStreamOverlay(frame, tracks, stats);
        if (streamer) {
            streamer->pushFrame(frame);
        }
    }
}
