#include "main.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <csignal>
#include <atomic>
#include <vector>
#include <algorithm>
#include <numeric>

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

// 计算图像清晰度（拉普拉斯方差）
double computeClarity(const Mat& img) {
    Mat gray, lap;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Laplacian(gray, lap, CV_64F);
    Scalar mean_val, stddev_val;
    meanStdDev(lap, mean_val, stddev_val);
    return stddev_val.val[0] * stddev_val.val[0];
}

int main(int argc, char** argv) {
    // 解析命令行参数
    bool save_images = false;
    int save_interval = 100;
    bool test_clarity = false;
    bool test_resize = false;
    
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--save") {
            save_images = true;
        } else if (arg == "--interval" && i + 1 < argc) {
            save_interval = atoi(argv[++i]);
        } else if (arg == "--clarity") {
            test_clarity = true;
        } else if (arg == "--resize") {
            test_resize = true;
        } else if (arg == "--help") {
            printf("用法: %s [选项]\n", argv[0]);
            printf("选项:\n");
            printf("  --save             每隔N帧保存一张图片\n");
            printf("  --interval N       设置保存间隔帧数（默认100）\n");
            printf("  --clarity          测试图像清晰度\n");
            printf("  --resize           测试缩放性能(4K->720p)\n");
            printf("  --help             显示此帮助信息\n");
            return 0;
        }
    }
    
    log_info("=== 摄像头帧测试程序 ===");
    log_info("分辨率: %dx%d (%dK)", CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_WIDTH/1920);
    if (save_images) log_info("图片保存: 每 %d 帧保存一次", save_interval);
    if (test_clarity) log_info("清晰度测试: 开启");
    if (test_resize) log_info("缩放测试: 开启 (4K->720p)");
    
    // 初始化摄像头
    if (mipicamera_init(CAMERA_INDEX_1, CAMERA_WIDTH, CAMERA_HEIGHT, 0) != 0) {
        log_error("摄像头初始化失败!");
        return -1;
    }
    mipicamera_set_format(CAMERA_INDEX_1, RK_FORMAT_RGB_888);

    log_info("摄像头初始化成功");
    
    // 设置信号处理
    std::signal(SIGINT, handleSignal);
    
    // 统计变量
    long total_frames = 0;
    auto start_time = std::chrono::steady_clock::now();
    auto last_fps_time = start_time;
    long frames_at_last_update = 0;
    double current_fps = 0.0;
    
    // 清晰度统计
    vector<double> clarity_history;
    double min_clarity = 999999.0;
    double max_clarity = 0.0;
    double avg_clarity = 0.0;
    
    // 性能统计
    long total_capture_time = 0;  // 微秒
    long total_resize_time = 0;   // 微秒
    long total_clarity_time = 0;  // 微秒
    
    vector<unsigned char> buffer(IMAGE_SIZE);
    
    log_info("开始连续拍摄，按 Ctrl+C 停止...");
    log_info("=================================");
    
    while (running) {
        auto frame_start = std::chrono::high_resolution_clock::now();
        
        // 获取一帧
        if (mipicamera_getframe(CAMERA_INDEX_1, reinterpret_cast<char*>(buffer.data())) != 0) {
            log_error("获取帧失败");
            continue;
        }
        
        auto capture_end = std::chrono::high_resolution_clock::now();
        total_capture_time += std::chrono::duration_cast<std::chrono::microseconds>(capture_end - frame_start).count();
        
        // 创建Mat对象
        Mat frame(CAMERA_HEIGHT, CAMERA_WIDTH, CV_8UC3, buffer.data());
        if (frame.empty()) {
            log_error("帧数据为空");
            continue;
        }
        
        // 帧计数
        total_frames++;
        
        // 缩放测试
        if (test_resize) {
            auto resize_start = std::chrono::high_resolution_clock::now();
            Mat resized;
            cv::resize(frame, resized, Size(IMAGE_WIDTH, IMAGE_HEIGHT), 0, 0, cv::INTER_LINEAR);
            auto resize_end = std::chrono::high_resolution_clock::now();
            total_resize_time += std::chrono::duration_cast<std::chrono::microseconds>(resize_end - resize_start).count();
        }
        
        // 清晰度测试
        if (test_clarity) {
            auto clarity_start = std::chrono::high_resolution_clock::now();
            double clarity = computeClarity(frame);
            auto clarity_end = std::chrono::high_resolution_clock::now();
            total_clarity_time += std::chrono::duration_cast<std::chrono::microseconds>(clarity_end - clarity_start).count();
            
            clarity_history.push_back(clarity);
            if (clarity < min_clarity) min_clarity = clarity;
            if (clarity > max_clarity) max_clarity = clarity;
            
            // 只保留最近100帧的清晰度数据
            if (clarity_history.size() > 100) {
                clarity_history.erase(clarity_history.begin());
            }
        }
        
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
            
            // 计算平均清晰度
            if (test_clarity && !clarity_history.empty()) {
                avg_clarity = std::accumulate(clarity_history.begin(), clarity_history.end(), 0.0) / clarity_history.size();
            }
            
            // 输出统计信息
            string stats = "帧统计: 总帧数=" + to_string(total_frames) + 
                          ", 当前FPS=" + to_string((int)current_fps) + 
                          ", 运行时间=" + to_string(total_elapsed.count()) + "s";
            
            if (test_clarity) {
                stats += ", 清晰度=" + to_string((int)avg_clarity);
            }
            
            log_info("%s", stats.c_str());
        }
        
        // 保存测试图片
        if (save_images && total_frames % save_interval == 0) {
            char filename[256];
            snprintf(filename, sizeof(filename), "frame_%06ld.jpg", total_frames);
            if (cv::imwrite(filename, frame)) {
                log_debug("已保存: %s", filename);
            }
        }
    }
    
    // 清理
    mipicamera_exit(CAMERA_INDEX_1);
    
    // 输出最终统计
    auto end_time = std::chrono::steady_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double avg_fps = total_time.count() > 0 ? (double)total_frames * 1000.0 / total_time.count() : 0.0;
    
    log_info("=================================");
    log_info("测试结束统计:");
    log_info("  总帧数: %ld", total_frames);
    log_info("  运行时间: %.2f 秒", total_time.count() / 1000.0);
    log_info("  平均FPS: %.2f", avg_fps);
    
    if (total_frames > 0) {
        log_info("  平均帧捕获耗时: %.2f ms", total_capture_time / 1000.0 / total_frames);
        
        if (test_resize) {
            log_info("  平均缩放耗时: %.2f ms", total_resize_time / 1000.0 / total_frames);
        }
        
        if (test_clarity) {
            log_info("  平均清晰度计算耗时: %.2f ms", total_clarity_time / 1000.0 / total_frames);
            log_info("  清晰度范围: %.2f - %.2f (平均: %.2f)", min_clarity, max_clarity, avg_clarity);
        }
    }
    
    log_info("=================================");
    
    return 0;
}
