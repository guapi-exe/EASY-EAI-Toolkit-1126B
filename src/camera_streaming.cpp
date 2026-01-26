#include "main.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <csignal>
#include <atomic>
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <string>
#include <thread>
#include <chrono>

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
    bool is_rtsp;  // 是否是 RTSP 模式
    // RTSP Server 相关
    GstRTSPServer *rtsp_server;
    GstRTSPMountPoints *mounts;
    GstRTSPMediaFactory *factory;
    // 用于复用的 Mat，避免重复分配
    Mat *scaled_mat;
    Mat *frame_mat;  // 复用摄像头帧的 Mat
    std::vector<unsigned char> *camera_buffer;  // 复用摄像头 buffer
    guint64 frame_count;  // 帧计数，用于追踪客户端连接
    guint64 dropped_frames;  // 丢帧计数
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
    gsize frame_size = width * height * 3;  // BGR
    g_object_set(G_OBJECT(ctx->appsrc),
                 "stream-type", 0,  // GST_APP_STREAM_TYPE_STREAM
                 "format", GST_FORMAT_TIME,
                 "is-live", TRUE,
                 "do-timestamp", TRUE,
                 "max-bytes", frame_size * 3,  // 最多缓冲 3 帧
                 "block", FALSE,  // 不阻塞
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

// RTSP media 配置回调
static void rtsp_media_configure(GstRTSPMediaFactory *factory, GstRTSPMedia *media, gpointer user_data) {
    StreamContext *ctx = (StreamContext *)user_data;
    
    log_info("RTSP client connected, configuring media...");
    
    GstElement *element = gst_rtsp_media_get_element(media);
    if (!element) {
        log_error("Failed to get media element");
        return;
    }
    
    // 查找 appsrc（使用 gst_bin_get_by_name）
    GstElement *appsrc = gst_bin_get_by_name(GST_BIN(element), "videosrc");
    
    if (appsrc) {
        log_info("Found appsrc, configuring...");
        
        // 配置 appsrc
        gst_util_set_object_arg(G_OBJECT(appsrc), "format", "time");
        
        // 计算单帧大小
        gsize frame_size = ctx->width * ctx->height * 3;  // BGR
        
        g_object_set(G_OBJECT(appsrc),
                     "is-live", TRUE,
                     "do-timestamp", TRUE,
                     "format", GST_FORMAT_TIME,
                     "max-bytes", frame_size * 3,  // 最多缓冲 3 帧，防止堆积
                     "block", FALSE,  // 不阻塞，缓冲区满时丢帧
                     "emit-signals", FALSE,
                     NULL);
        
        // 设置 caps
        GstCaps *caps = gst_caps_new_simple("video/x-raw",
                                            "format", G_TYPE_STRING, "BGR",
                                            "width", G_TYPE_INT, ctx->width,
                                            "height", G_TYPE_INT, ctx->height,
                                            "framerate", GST_TYPE_FRACTION, ctx->fps, 1,
                                            NULL);
        g_object_set(G_OBJECT(appsrc), "caps", caps, NULL);
        gst_caps_unref(caps);
        
        // 保存 appsrc 引用，如果已有旧引用则先释放
        if (ctx->appsrc) {
            log_info("Releasing old appsrc reference");
            gst_object_unref(ctx->appsrc);
            ctx->appsrc = NULL;
        }
        
        ctx->appsrc = (GstElement *)gst_object_ref(appsrc);
        ctx->frame_count = 0;  // 重置帧计数
        log_info("Client connected! Ready to stream at %dx%d@%dfps", 
                 ctx->width, ctx->height, ctx->fps);
        
        gst_object_unref(appsrc);
    } else {
        log_error("Failed to find appsrc element 'videosrc'");
    }
    
    if (element) {
        gst_object_unref(element);
    }
}

// 创建 RTSP 推流 pipeline
bool create_rtsp_pipeline(StreamContext *ctx, const char *rtsp_url, int width, int height, int fps) {
    ctx->width = width;
    ctx->height = height;
    ctx->fps = fps;
    ctx->is_rtsp = true;
    
    // 解析 RTSP URL (格式: rtsp://host:port/path)
    const char *port = "8554";
    const char *mount_path = "/stream";
    
    if (rtsp_url && strlen(rtsp_url) > 0) {
        // 简单解析 URL，提取端口和路径
        const char *port_start = strstr(rtsp_url, ":");
        if (port_start) {
            port_start = strchr(port_start + 3, ':'); // 跳过 "://"
            if (port_start) {
                port = port_start + 1;
            }
        }
        const char *path_start = strrchr(rtsp_url, '/');
        if (path_start && strlen(path_start) > 1) {
            mount_path = path_start;
        }
    }
    
    // 创建 RTSP Server
    ctx->rtsp_server = gst_rtsp_server_new();
    if (!ctx->rtsp_server) {
        log_error("Failed to create RTSP server");
        return false;
    }
    
    // 设置端口
    g_object_set(ctx->rtsp_server, "service", port, NULL);
    
    // 获取 mount points
    ctx->mounts = gst_rtsp_server_get_mount_points(ctx->rtsp_server);
    
    // 创建 media factory
    ctx->factory = gst_rtsp_media_factory_new();
    if (!ctx->factory) {
        log_error("Failed to create RTSP media factory");
        return false;
    }
    
    // 设置 pipeline 描述（使用 appsrc）
    char launch_str[512];
    snprintf(launch_str, sizeof(launch_str),
             "appsrc name=videosrc format=time ! "
             "videoconvert ! "
             "x264enc tune=zerolatency bitrate=2000 speed-preset=superfast key-int-max=%d ! "
             "rtph264pay name=pay0 pt=96",
             fps * 2);
    
    gst_rtsp_media_factory_set_launch(ctx->factory, launch_str);
    gst_rtsp_media_factory_set_shared(ctx->factory, TRUE);
    
    // 连接 media-configure 信号
    g_signal_connect(ctx->factory, "media-configure",
                     G_CALLBACK(rtsp_media_configure), ctx);
    
    // 添加到 mount points
    gst_rtsp_mount_points_add_factory(ctx->mounts, mount_path, ctx->factory);
    g_object_unref(ctx->mounts);
    
    // attach server
    gst_rtsp_server_attach(ctx->rtsp_server, NULL);
    
    log_info("RTSP server created");
    log_info("Stream ready at rtsp://localhost:%s%s", port, mount_path);
    
    return true;
}

// 推送视频帧到 GStreamer
bool push_frame(StreamContext *ctx, const Mat &frame) {
    if (!ctx) {
        return false;
    }
    
    // RTSP 模式下，如果没有客户端连接，appsrc 可能为 NULL，这是正常的
    if (!ctx->appsrc) {
        if (ctx->is_rtsp) {
            // RTSP 模式，等待客户端连接，不报错
            return true;
        }
        return false;
    }
    
    // 首次推送时记录日志
    if (ctx->frame_count == 0 && ctx->is_rtsp) {
        log_info("Starting to push frames to client...");
    }
    ctx->frame_count++;
    
    // 调整图像大小（如果需要），使用持久 Mat 避免重复分配
    Mat *scaled_frame_ptr = nullptr;
    if (frame.cols != ctx->width || frame.rows != ctx->height) {
        if (!ctx->scaled_mat) {
            ctx->scaled_mat = new Mat();
        }
        cv::resize(frame, *ctx->scaled_mat, Size(ctx->width, ctx->height));
        scaled_frame_ptr = ctx->scaled_mat;
    } else {
        scaled_frame_ptr = const_cast<Mat*>(&frame);
    }
    
    // 创建 GstBuffer
    gsize size = scaled_frame_ptr->total() * scaled_frame_ptr->elemSize();
    GstBuffer *buffer = gst_buffer_new_allocate(NULL, size, NULL);
    
    if (!buffer) {
        log_error("Failed to allocate GstBuffer");
        return false;
    }
    
    // 填充数据
    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
        log_error("Failed to map GstBuffer");
        gst_buffer_unref(buffer);
        return false;
    }
    
    memcpy(map.data, scaled_frame_ptr->data, size);
    gst_buffer_unmap(buffer, &map);
    
    // 推送到 appsrc
    GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(ctx->appsrc), buffer);
    
    if (ret != GST_FLOW_OK) {
        // buffer 已经被 appsrc 接管或释放，不需要手动 unref
        if (ret == GST_FLOW_FLUSHING) {
            // 客户端可能已断开
            log_warn("appsrc is flushing, client may have disconnected");
            if (ctx->is_rtsp && ctx->appsrc) {
                gst_object_unref(ctx->appsrc);
                ctx->appsrc = NULL;
            }
            return false;
        } else if (ret == GST_FLOW_EOS) {
            log_warn("appsrc reached EOS");
            return false;
        } else {
            // 其他错误，可能是缓冲区满，丢弃这帧
            ctx->dropped_frames++;
            if (ctx->dropped_frames % 100 == 1) {  // 每 100 帧报告一次
                log_warn("Dropped frames: %llu (buffer may be full)", ctx->dropped_frames);
            }
            return true;  // 返回 true 继续运行
        }
    }
    
    return true;
}

// 启动推流
bool start_streaming(StreamContext *ctx) {
    if (!ctx) {
        return false;
    }
    
    // RTSP 模式不需要启动 pipeline（由 RTSP server 管理）
    if (ctx->rtsp_server) {
        log_info("RTSP server started, waiting for connections...");
        log_info("Use this command to view stream:");
        log_info("  ffplay rtsp://localhost:8554/stream");
        log_info("  or vlc rtsp://localhost:8554/stream");
        return true;
    }
    
    // RTMP 模式需要启动 pipeline
    if (!ctx->pipeline) {
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
        gst_object_unref(ctx->appsrc);
        ctx->appsrc = NULL;
    }
    
    if (ctx->pipeline) {
        gst_element_set_state(ctx->pipeline, GST_STATE_NULL);
        gst_object_unref(ctx->pipeline);
        ctx->pipeline = NULL;
    }
    
    if (ctx->rtsp_server) {
        g_object_unref(ctx->rtsp_server);
        ctx->rtsp_server = NULL;
    }
    
    if (ctx->loop) {
        g_main_loop_quit(ctx->loop);
        g_main_loop_unref(ctx->loop);
        ctx->loop = NULL;
    }
    
    // 清理 scaled_mat
    if (ctx->scaled_mat) {
        delete ctx->scaled_mat;
        ctx->scaled_mat = NULL;
    }
    
    // 清理 frame_mat
    if (ctx->frame_mat) {
        delete ctx->frame_mat;
        ctx->frame_mat = NULL;
    }
    
    // 清理 camera_buffer
    if (ctx->camera_buffer) {
        delete ctx->camera_buffer;
        ctx->camera_buffer = NULL;
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
    
    // 初始化复用的 buffer 和 Mat
    if (!stream_ctx.camera_buffer) {
        stream_ctx.camera_buffer = new vector<unsigned char>(IMAGE_SIZE);
    }
    if (!stream_ctx.frame_mat) {
        stream_ctx.frame_mat = new Mat(CAMERA_HEIGHT, CAMERA_WIDTH, CV_8UC3);
    }
    
    // 获取默认 GMainContext（用于 RTSP Server）
    GMainContext *main_context = g_main_context_default();
    
    log_info("开始推流，按 Ctrl+C 停止...");
    log_info("=================================");
    
    while (running) {
        // 处理 GMainContext 事件（RTSP Server 需要）
        // 限制每次最多处理 10 个事件，避免阻塞太久
        int max_iterations = 10;
        while (max_iterations-- > 0 && g_main_context_iteration(main_context, FALSE)) {
            // 处理待处理的事件
        }
        
        // 获取一帧
        if (mipicamera_getframe(CAMERA_INDEX_1, reinterpret_cast<char*>(stream_ctx.camera_buffer->data())) != 0) {
            log_error("获取帧失败");
            continue;
        }
        
        // 使用复用的 Mat，不复制数据，只是包装 buffer
        stream_ctx.frame_mat->data = stream_ctx.camera_buffer->data();
        
        if (stream_ctx.frame_mat->empty()) {
            log_error("帧数据为空");
            continue;
        }
        
        // 推送到流
        if (!push_frame(&stream_ctx, *stream_ctx.frame_mat)) {
            // 不再 break，继续运行（可能只是客户端断开）
            // log_error("推送帧失败");
            // break;
        }
        
        // 保存到本地（可选）
        if (save_local && video_writer.isOpened()) {
            Mat resized_frame;
            cv::resize(frame, resized_frame, Size(output_width, output_height));
            video_writer.write(resized_frame);
        }
        
        total_frames++;
        
        // 每 300 帧（约 10 秒）强制清理一次
        if (total_frames % 300 == 0) {
            // 强制触发垃圾回收
            if (stream_ctx.scaled_mat && !stream_ctx.scaled_mat->empty()) {
                stream_ctx.scaled_mat->release();
            }
            // 清理 GStreamer 缓存
            g_main_context_wakeup(main_context);
        }
        
        // 计算 FPS（每秒更新一次）
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_fps_time);
        
        if (elapsed.count() >= 1) {
            long frames_diff = total_frames - frames_at_last_update;
            current_fps = static_cast<double>(frames_diff) / elapsed.count();
            
            frames_at_last_update = total_frames;
            last_fps_time = now;
            
            auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
            
            // 读取内存使用情况（仅 Linux）
            long mem_kb = 0;
            FILE* fp = fopen("/proc/self/status", "r");
            if (fp) {
                char line[128];
                while (fgets(line, sizeof(line), fp)) {
                    if (strncmp(line, "VmRSS:", 6) == 0) {
                        sscanf(line + 6, "%ld", &mem_kb);
                        break;
                    }
                }
                fclose(fp);
            }
            
            if (mem_kb > 0) {
                log_info("推流统计: 总帧数=%ld, 当前FPS=%.2f, 运行时间=%lds, 内存=%.2fMB, 丢帧=%llu", 
                         total_frames, current_fps, total_elapsed.count(), mem_kb / 1024.0, stream_ctx.dropped_frames);
            } else {
                log_info("推流统计: 总帧数=%ld, 当前FPS=%.2f, 运行时间=%lds, 丢帧=%llu", 
                         total_frames, current_fps, total_elapsed.count(), stream_ctx.dropped_frames);
            }
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
