#include "main.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <csignal>
#include <atomic>

extern "C" {
#include "log.h"
#include "camera.h"
}

using namespace cv;
using namespace std;

std::atomic<bool> running(true);

void handleSignal(int) {
    running = false;
}

int main() {
    log_info("=== 摄像头帧测试程序 ===");
    log_info("分辨率: %dx%d", CAMERA_WIDTH, CAMERA_HEIGHT);
    
    // 初始化摄像头
    if (mipicamera_init(CAMERA_INDEX_1, CAMERA_WIDTH, CAMERA_HEIGHT, 0) != 0) {
        log_error("摄像头初始化失败!");
        return -1;
    }
    log_info("摄像头初始化成功");
    
    // 设置信号处理
    std::signal(SIGINT, handleSignal);
    
    // 帧计数和时间统计
    long total_frames = 0;
    auto start_time = std::chrono::steady_clock::now();
    auto last_fps_time = start_time;
    long frames_at_last_update = 0;
    double current_fps = 0.0;
    
    vector<unsigned char> buffer(IMAGE_SIZE);
    
    log_info("开始连续拍摄，按 Ctrl+C 停止...");
    log_info("=================================");
    
    while (running) {
        // 获取一帧
        if (mipicamera_getframe(CAMERA_INDEX_1, reinterpret_cast<char*>(buffer.data())) != 0) {
            log_error("获取帧失败");
            continue;
        }
        
        // 创建Mat对象
        Mat frame(CAMERA_HEIGHT, CAMERA_WIDTH, CV_8UC3, buffer.data());
        if (frame.empty()) {
            log_error("帧数据为空");
            continue;
        }
        
        // 帧计数
        total_frames++;
        
        // 计算FPS（每秒更新一次）
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_fps_time);
        
        if (elapsed.count() >= 1) {
            long frames_diff = total_frames - frames_at_last_update;
            current_fps = static_cast<double>(frames_diff) / elapsed.count();
            
            frames_at_last_update = total_frames;
            last_fps_time = now;
            
            // 计算总运行时间
            auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
            
            // 输出统计信息
            log_info("帧统计: 总帧数=%ld, 当前FPS=%.2f, 运行时间=%lds", 
                     total_frames, current_fps, total_elapsed.count());
        }
        
        // 每100帧保存一张测试图片
        if (total_frames % 100 == 0) {
            char filename[256];
            snprintf(filename, sizeof(filename), "test_frame_%ld.jpg", total_frames);
            if (cv::imwrite(filename, frame)) {
                log_debug("已保存测试图片: %s", filename);
            }
        }
        
        // 可选：添加短暂延迟避免CPU占用过高（根据需要调整或删除）
        // std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    // 清理
    mipicamera_exit(CAMERA_INDEX_1);
    
    // 输出最终统计
    auto end_time = std::chrono::steady_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    double avg_fps = total_time.count() > 0 ? (double)total_frames / total_time.count() : 0.0;
    
    log_info("=================================");
    log_info("测试结束统计:");
    log_info("  总帧数: %ld", total_frames);
    log_info("  运行时间: %ld 秒", total_time.count());
    log_info("  平均FPS: %.2f", avg_fps);
    log_info("=================================");
    
    return 0;
}
