#include "main.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <csignal>
#include <atomic>
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <string>

extern "C" {
#include "log.h"
#include "camera.h"
}

using namespace cv;
using namespace std;

std::atomic<bool> running(true);

// GStreamer pipeline 相关
typedef struct {
    GstElement *pipeline;
    GstElement *appsrc;
    GstElement *videoconvert;
    GstElement *videoscale;
    GstElement *encoder;
    GstElement *muxer;
    GstElement *sink;
    GMainLoop *loop;
    guint fps;
    gint width;
    gint height;
} StreamContext;

void handleSignal(int) {
    running = false;
}

// GStreamer 总线消息处理
static gboolean bus_callback(GstBus *bus, GstMessage *msg, gpointer data) {
    GMainLoop *loop = (GMainLoop *)data;
    
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_ERROR: {
            GError *err;
            gchar *debug;
            gst_message_parse_error(msg, &err, &debug);
            log_error("GStreamer Error: %s", err->message);
            g_error_free(err);
            g_free(debug);
            g_main_loop_quit(loop);
            break;
        }
        case GST_MESSAGE_WARNING: {
            GError *err;
            gchar *debug;
            gst_message_parse_warning(msg, &err, &debug);
            log_warn("GStreamer Warning: %s", err->message);
            g_error_free(err);
            g_free(debug);
            break;
        }
        case GST_MESSAGE_EOS:
            log_info("End-Of-Stream reached");
            g_main_loop_quit(loop);
            break;
        default:
            break;
    }
    return TRUE;
}

// 创建 RTMP 推流 pipeline
bool create_rtmp_pipeline(StreamContext *ctx, const char *rtmp_url, int width, int height, int fps) {
    ctx->width = width;
    ctx->height = height;
    ctx->fps = fps;
    
    // 创建元素
    ctx->pipeline = gst_pipeline_new("rtmp-pipeline");
    ctx->appsrc = gst_element_factory_make("appsrc", "source");
    ctx->videoconvert = gst_element_factory_make("videoconvert", "convert");
    ctx->videoscale = gst_element_factory_make("videoscale", "scale");
    ctx->encoder = gst_element_factory_make("x264enc", "encoder");
    ctx->muxer = gst_element_factory_make("flvmux", "muxer");
    ctx->sink = gst_element_factory_make("rtmpsink", "sink");
    
    if (!ctx->pipeline || !ctx->appsrc || !ctx->videoconvert || 
        !ctx->videoscale || !ctx->encoder || !ctx->muxer || !ctx->sink) {
        log_error("Failed to create GStreamer elements for RTMP");
        return false;
    }
    
    // 配置 appsrc
    g_object_set(G_OBJECT(ctx->appsrc),
                 "stream-type", 0,  // GST_APP_STREAM_TYPE_STREAM
                 "format", GST_FORMAT_TIME,
                 "is-live", TRUE,
                 "do-timestamp", TRUE,
                 NULL);
    
    // 设置 caps
    char caps_str[256];
    snprintf(caps_str, sizeof(caps_str),
             "video/x-raw,format=BGR,width=%d,height=%d,framerate=%d/1",
             width, height, fps);
    GstCaps *caps = gst_caps_from_string(caps_str);
    gst_app_src_set_caps(GST_APP_SRC(ctx->appsrc), caps);
    gst_caps_unref(caps);
    
    // 配置编码器（低延迟配置）
    g_object_set(G_OBJECT(ctx->encoder),
                 "tune", 0x00000004,  // zerolatency
                 "bitrate", 2000,     // 2000 kbps
                 "speed-preset", 1,   // superfast
                 "key-int-max", fps * 2,  // 每2秒一个关键帧
                 NULL);
    
    // 设置 RTMP 地址
    g_object_set(G_OBJECT(ctx->sink),
                 "location", rtmp_url,
                 NULL);
    
    // 添加所有元素到 pipeline
    gst_bin_add_many(GST_BIN(ctx->pipeline),
                     ctx->appsrc, ctx->videoconvert, ctx->videoscale,
                     ctx->encoder, ctx->muxer, ctx->sink, NULL);
    
    // 链接元素
    if (!gst_element_link_many(ctx->appsrc, ctx->videoconvert, ctx->videoscale,
                                ctx->encoder, ctx->muxer, ctx->sink, NULL)) {
        log_error("Failed to link GStreamer elements for RTMP");
        return false;
    }
    
    // 创建主循环
    ctx->loop = g_main_loop_new(NULL, FALSE);
    
    // 添加总线监听
    GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(ctx->pipeline));
    gst_bus_add_watch(bus, bus_callback, ctx->loop);
    gst_object_unref(bus);
    
    log_info("RTMP pipeline created: %s", rtmp_url);
    return true;
}

// 创建 RTSP 推流 pipeline
bool create_rtsp_pipeline(StreamContext *ctx, const char *rtsp_url, int width, int height, int fps) {
    ctx->width = width;
    ctx->height = height;
    ctx->fps = fps;
    
    // 创建元素
    ctx->pipeline = gst_pipeline_new("rtsp-pipeline");
    ctx->appsrc = gst_element_factory_make("appsrc", "source");
    ctx->videoconvert = gst_element_factory_make("videoconvert", "convert");
    ctx->videoscale = gst_element_factory_make("videoscale", "scale");
    ctx->encoder = gst_element_factory_make("x264enc", "encoder");
    ctx->muxer = gst_element_factory_make("rtph264pay", "payloader");
    ctx->sink = gst_element_factory_make("udpsink", "sink");
    
    if (!ctx->pipeline || !ctx->appsrc || !ctx->videoconvert || 
        !ctx->videoscale || !ctx->encoder || !ctx->muxer || !ctx->sink) {
        log_error("Failed to create GStreamer elements for RTSP");
        return false;
    }
    
    // 配置 appsrc
    g_object_set(G_OBJECT(ctx->appsrc),
                 "stream-type", 0,
                 "format", GST_FORMAT_TIME,
                 "is-live", TRUE,
                 "do-timestamp", TRUE,
                 NULL);
    
    // 设置 caps
    char caps_str[256];
    snprintf(caps_str, sizeof(caps_str),
             "video/x-raw,format=BGR,width=%d,height=%d,framerate=%d/1",
             width, height, fps);
    GstCaps *caps = gst_caps_from_string(caps_str);
    gst_app_src_set_caps(GST_APP_SRC(ctx->appsrc), caps);
    gst_caps_unref(caps);
    
    // 配置编码器
    g_object_set(G_OBJECT(ctx->encoder),
                 "tune", 0x00000004,
                 "bitrate", 2000,
                 "speed-preset", 1,
                 NULL);
    
    // 配置 UDP sink（RTSP 通常使用 UDP 传输）
    g_object_set(G_OBJECT(ctx->sink),
                 "host", "127.0.0.1",
                 "port", 5000,
                 NULL);
    
    // 添加元素到 pipeline
    gst_bin_add_many(GST_BIN(ctx->pipeline),
                     ctx->appsrc, ctx->videoconvert, ctx->videoscale,
                     ctx->encoder, ctx->muxer, ctx->sink, NULL);
    
    // 链接元素
    if (!gst_element_link_many(ctx->appsrc, ctx->videoconvert, ctx->videoscale,
                                ctx->encoder, ctx->muxer, ctx->sink, NULL)) {
        log_error("Failed to link GStreamer elements for RTSP");
        return false;
    }
    
    // 创建主循环
    ctx->loop = g_main_loop_new(NULL, FALSE);
    
    // 添加总线监听
    GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(ctx->pipeline));
    gst_bus_add_watch(bus, bus_callback, ctx->loop);
    gst_object_unref(bus);
    
    log_info("RTSP pipeline created");
    return true;
}

// 推送视频帧到 GStreamer
bool push_frame(StreamContext *ctx, const Mat &frame) {
    if (!ctx || !ctx->appsrc) {
        return false;
    }
    
    // 调整图像大小（如果需要）
    Mat scaled_frame;
    if (frame.cols != ctx->width || frame.rows != ctx->height) {
        cv::resize(frame, scaled_frame, Size(ctx->width, ctx->height));
    } else {
        scaled_frame = frame;
    }
    
    // 创建 GstBuffer
    gsize size = scaled_frame.total() * scaled_frame.elemSize();
    GstBuffer *buffer = gst_buffer_new_allocate(NULL, size, NULL);
    
    // 填充数据
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_WRITE);
    memcpy(map.data, scaled_frame.data, size);
    gst_buffer_unmap(buffer, &map);
    
    // 推送到 appsrc
    GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(ctx->appsrc), buffer);
    
    if (ret != GST_FLOW_OK) {
        log_error("Failed to push buffer to appsrc");
        return false;
    }
    
    return true;
}

// 启动推流
bool start_streaming(StreamContext *ctx) {
    if (!ctx || !ctx->pipeline) {
        return false;
    }
    
    GstStateChangeReturn ret = gst_element_set_state(ctx->pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        log_error("Failed to start pipeline");
        return false;
    }
    
    log_info("Streaming started");
    return true;
}

// 停止推流
void stop_streaming(StreamContext *ctx) {
    if (!ctx) return;
    
    if (ctx->appsrc) {
        gst_app_src_end_of_stream(GST_APP_SRC(ctx->appsrc));
    }
    
    if (ctx->pipeline) {
        gst_element_set_state(ctx->pipeline, GST_STATE_NULL);
        gst_object_unref(ctx->pipeline);
    }
    
    if (ctx->loop) {
        g_main_loop_quit(ctx->loop);
        g_main_loop_unref(ctx->loop);
    }
    
    log_info("Streaming stopped");
}

void print_usage(const char *prog_name) {
    printf("用法: %s [选项]\n", prog_name);
    printf("\n选项:\n");
    printf("  --rtmp URL         RTMP 推流地址 (例如: rtmp://localhost/live/stream)\n");
    printf("  --rtsp             使用 RTSP 推流（UDP端口5000）\n");
    printf("  --width WIDTH      输出宽度（默认: 1280）\n");
    printf("  --height HEIGHT    输出高度（默认: 720）\n");
    printf("  --fps FPS          输出帧率（默认: 25）\n");
    printf("  --save             保存本地视频文件\n");
    printf("  --help             显示此帮助信息\n");
    printf("\n示例:\n");
    printf("  %s --rtmp rtmp://localhost/live/stream\n", prog_name);
    printf("  %s --rtsp --width 1920 --height 1080 --fps 30\n", prog_name);
}

int main(int argc, char** argv) {
    // 解析命令行参数
    string rtmp_url = "";
    bool use_rtsp = false;
    int output_width = 1280;
    int output_height = 720;
    int output_fps = 25;
    bool save_local = false;
    
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--rtmp" && i + 1 < argc) {
            rtmp_url = argv[++i];
        } else if (arg == "--rtsp") {
            use_rtsp = true;
        } else if (arg == "--width" && i + 1 < argc) {
            output_width = atoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            output_height = atoi(argv[++i]);
        } else if (arg == "--fps" && i + 1 < argc) {
            output_fps = atoi(argv[++i]);
        } else if (arg == "--save") {
            save_local = true;
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    if (rtmp_url.empty() && !use_rtsp) {
        log_error("请指定推流方式: --rtmp 或 --rtsp");
        print_usage(argv[0]);
        return -1;
    }
    
    log_info("=== 摄像头推流程序 ===");
    log_info("摄像头分辨率: %dx%d", CAMERA_WIDTH, CAMERA_HEIGHT);
    log_info("输出分辨率: %dx%d @ %dfps", output_width, output_height, output_fps);
    
    if (!rtmp_url.empty()) {
        log_info("推流方式: RTMP");
        log_info("推流地址: %s", rtmp_url.c_str());
    } else if (use_rtsp) {
        log_info("推流方式: RTSP (UDP:5000)");
    }
    
    // 初始化 GStreamer
    gst_init(&argc, &argv);
    
    // 初始化摄像头
    if (mipicamera_init(CAMERA_INDEX_1, CAMERA_WIDTH, CAMERA_HEIGHT, 0) != 0) {
        log_error("摄像头初始化失败!");
        return -1;
    }
    mipicamera_set_format(CAMERA_INDEX_1, RK_FORMAT_RGB_888);
    log_info("摄像头初始化成功");
    
    // 创建推流上下文
    StreamContext stream_ctx = {0};
    bool pipeline_ok = false;
    
    if (!rtmp_url.empty()) {
        pipeline_ok = create_rtmp_pipeline(&stream_ctx, rtmp_url.c_str(), 
                                           output_width, output_height, output_fps);
    } else if (use_rtsp) {
        pipeline_ok = create_rtsp_pipeline(&stream_ctx, NULL,
                                           output_width, output_height, output_fps);
    }
    
    if (!pipeline_ok) {
        log_error("创建推流 pipeline 失败!");
        mipicamera_exit(CAMERA_INDEX_1);
        return -1;
    }
    
    // 启动推流
    if (!start_streaming(&stream_ctx)) {
        log_error("启动推流失败!");
        stop_streaming(&stream_ctx);
        mipicamera_exit(CAMERA_INDEX_1);
        return -1;
    }
    
    // 设置信号处理
    std::signal(SIGINT, handleSignal);
    
    // 本地视频保存（可选）
    VideoWriter video_writer;
    if (save_local) {
        string filename = "output_" + to_string(time(NULL)) + ".mp4";
        video_writer.open(filename, VideoWriter::fourcc('H','2','6','4'),
                         output_fps, Size(output_width, output_height));
        if (video_writer.isOpened()) {
            log_info("本地保存: %s", filename.c_str());
        }
    }
    
    // 统计变量
    long total_frames = 0;
    auto start_time = std::chrono::steady_clock::now();
    auto last_fps_time = start_time;
    long frames_at_last_update = 0;
    double current_fps = 0.0;
    
    vector<unsigned char> buffer(IMAGE_SIZE);
    
    log_info("开始推流，按 Ctrl+C 停止...");
    log_info("=================================");
    
    while (running) {
        // 获取一帧
        if (mipicamera_getframe(CAMERA_INDEX_1, reinterpret_cast<char*>(buffer.data())) != 0) {
            log_error("获取帧失败");
            continue;
        }
        
        // 创建 Mat 对象
        Mat frame(CAMERA_HEIGHT, CAMERA_WIDTH, CV_8UC3, buffer.data());
        if (frame.empty()) {
            log_error("帧数据为空");
            continue;
        }
        
        // 推送到流
        if (!push_frame(&stream_ctx, frame)) {
            log_error("推送帧失败");
            break;
        }
        
        // 保存到本地（可选）
        if (save_local && video_writer.isOpened()) {
            Mat resized_frame;
            cv::resize(frame, resized_frame, Size(output_width, output_height));
            video_writer.write(resized_frame);
        }
        
        total_frames++;
        
        // 计算 FPS（每秒更新一次）
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_fps_time);
        
        if (elapsed.count() >= 1) {
            long frames_diff = total_frames - frames_at_last_update;
            current_fps = static_cast<double>(frames_diff) / elapsed.count();
            
            frames_at_last_update = total_frames;
            last_fps_time = now;
            
            auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
            
            log_info("推流统计: 总帧数=%ld, 当前FPS=%.2f, 运行时间=%lds", 
                     total_frames, current_fps, total_elapsed.count());
        }
        
        // 控制帧率
        std::this_thread::sleep_for(std::chrono::milliseconds(1000 / output_fps));
    }
    
    // 清理
    if (video_writer.isOpened()) {
        video_writer.release();
    }
    
    stop_streaming(&stream_ctx);
    mipicamera_exit(CAMERA_INDEX_1);
    
    // 输出最终统计
    auto end_time = std::chrono::steady_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    double avg_fps = total_time.count() > 0 ? (double)total_frames / total_time.count() : 0.0;
    
    log_info("=================================");
    log_info("推流结束统计:");
    log_info("  总帧数: %ld", total_frames);
    log_info("  运行时间: %ld 秒", total_time.count());
    log_info("  平均FPS: %.2f", avg_fps);
    log_info("=================================");
    
    return 0;
}
