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

#include "face_detect_retian.h"

using namespace cv;
using namespace std;

std::atomic<bool> running(true);

void handleSignal(int) {
    running = false;
}

// 在图像上绘制检测结果
void drawDetections(Mat& img, const vector<RetinaFaceResult>& results) {
    // 颜色数组
    static Scalar colors[] = {
        Scalar(255, 0, 0),     // 蓝色
        Scalar(0, 255, 0),     // 绿色
        Scalar(0, 0, 255),     // 红色
        Scalar(255, 255, 0),   // 青色
        Scalar(255, 0, 255),   // 品红
        Scalar(0, 255, 255),   // 黄色
    };
    
    for (size_t i = 0; i < results.size(); i++) {
        const RetinaFaceResult& face = results[i];
        Scalar color = colors[i % 6];
        
        // 绘制边界框
        int x1 = (int)face.box.x;
        int y1 = (int)face.box.y;
        int x2 = (int)face.box.br().x;
        int y2 = (int)face.box.br().y;
        
        rectangle(img, Point(x1, y1), Point(x2, y2), color, 3);
        
        // 绘制置信度
        char text[64];
        sprintf(text, "Face%.0f%%", face.score * 100);
        int baseline = 0;
        Size textSize = getTextSize(text, FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
        rectangle(img, Point(x1, y1 - textSize.height - 8), 
                  Point(x1 + textSize.width, y1), color, -1);
        putText(img, text, Point(x1, y1 - 5), FONT_HERSHEY_SIMPLEX, 
                0.6, Scalar(255, 255, 255), 2);
        
        // 绘制关键点（左眼、右眼、鼻尖、左嘴角、右嘴角）
        static Scalar landmark_colors[] = {
            Scalar(0, 255, 255),   // 左眼 - 黄色
            Scalar(0, 255, 255),   // 右眼 - 黄色
            Scalar(255, 0, 255),   // 鼻尖 - 品红
            Scalar(0, 255, 0),     // 左嘴角 - 绿色
            Scalar(0, 255, 0),     // 右嘴角 - 绿色
        };
        
        for (size_t j = 0; j < face.landmarks.size(); j++) {
            circle(img, Point((int)face.landmarks[j].x, (int)face.landmarks[j].y), 
                   4, landmark_colors[j], -1);
        }
    }
}

int main(int argc, char** argv) {
    // 解析命令行参数
    string model_path = "./retinaface_480x640.rknn";  // 默认模型路径
    int model_type_int = 0;  // 默认 RETINAFACE
    bool save_images = false;
    int save_interval = 50;
    float conf_thresh = 0.5f;
    float nms_thresh = 0.4f;
    
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--type" && i + 1 < argc) {
            model_type_int = atoi(argv[++i]);
        } else if (arg == "--save") {
            save_images = true;
        } else if (arg == "--interval" && i + 1 < argc) {
            save_interval = atoi(argv[++i]);
        } else if (arg == "--conf" && i + 1 < argc) {
            conf_thresh = atof(argv[++i]);
        } else if (arg == "--nms" && i + 1 < argc) {
            nms_thresh = atof(argv[++i]);
        } else if (arg == "--help") {
            printf("用法: %s [选项]\n", argv[0]);
            printf("选项:\n");
            printf("  --model <path>     指定.rknn模型路径（默认：./retinaface_480x640.rknn）\n");
            printf("  --type <0|1|2>     模型类型：0=RETINAFACE, 1=SLIM, 2=RFB（默认：0）\n");
            printf("  --save             保存检测结果图片\n");
            printf("  --interval N       保存间隔帧数（默认50）\n");
            printf("  --conf <0.0-1.0>   置信度阈值（默认0.5）\n");
            printf("  --nms <0.0-1.0>    NMS阈值（默认0.4）\n");
            printf("  --help             显示此帮助信息\n");
            printf("\n示例:\n");
            printf("  %s --model ./rfb_480x640.rknn --type 2 --save --conf 0.6\n", argv[0]);
            return 0;
        }
    }
    
    RetinaFaceModelType model_type = RETINAFACE_MODEL;
    if (model_type_int == 1) model_type = SLIM_MODEL;
    else if (model_type_int == 2) model_type = RFB_MODEL;
    
    log_info("=== RetinaFace 人脸检测测试 ===");
    log_info("摄像头分辨率: %dx%d (%dK)", CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_WIDTH/1920);
    log_info("模型路径: %s", model_path.c_str());
    log_info("模型类型: %s", model_type == RETINAFACE_MODEL ? "RETINAFACE" : 
             (model_type == SLIM_MODEL ? "SLIM" : "RFB"));
    log_info("置信度阈值: %.2f", conf_thresh);
    log_info("NMS阈值: %.2f", nms_thresh);
    if (save_images) log_info("图片保存: 每 %d 帧保存一次", save_interval);
    
    // 初始化摄像头
    if (mipicamera_init(CAMERA_INDEX_1, CAMERA_WIDTH, CAMERA_HEIGHT, 0) != 0) {
        log_error("摄像头初始化失败!");
        return -1;
    }
    log_info("摄像头初始化成功");
    
    // 初始化 RetinaFace 模型
    RetinaFaceConfig config = get_retian_config(model_type, 480, 640);
    rknn_context ctx;
    if (face_detect_retian_init(&ctx, model_path.c_str(), &config) != 0) {
        log_error("RetinaFace 模型初始化失败!");
        mipicamera_exit(CAMERA_INDEX_1);
        return -1;
    }
    log_info("RetinaFace 模型初始化成功");
    
    // 设置信号处理
    std::signal(SIGINT, handleSignal);
    
    // 统计变量
    long total_frames = 0;
    long total_faces = 0;
    auto start_time = std::chrono::steady_clock::now();
    auto last_fps_time = start_time;
    long frames_at_last_update = 0;
    double current_fps = 0.0;
    
    // 性能统计
    long total_capture_time = 0;  // 微秒
    long total_resize_time = 0;   // 微秒
    long total_detect_time = 0;   // 微秒
    long total_draw_time = 0;     // 微秒
    
    vector<int> faces_per_frame;
    
    vector<unsigned char> buffer(IMAGE_SIZE);
    
    log_info("开始检测，按 Ctrl+C 停止...");
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
        
        // 创建Mat对象（4K原图）
        Mat frame(CAMERA_HEIGHT, CAMERA_WIDTH, CV_8UC3, buffer.data());
        if (frame.empty()) {
            log_error("帧数据为空");
            continue;
        }
        
        total_frames++;
        
        // 缩放到720p用于检测（可选，根据模型输入调整）
        auto resize_start = std::chrono::high_resolution_clock::now();
        Mat resized;
        cv::resize(frame, resized, Size(IMAGE_WIDTH, IMAGE_HEIGHT), 0, 0, cv::INTER_LINEAR);
        auto resize_end = std::chrono::high_resolution_clock::now();
        total_resize_time += std::chrono::duration_cast<std::chrono::microseconds>(resize_end - resize_start).count();
        
        // 人脸检测
        auto detect_start = std::chrono::high_resolution_clock::now();
        vector<RetinaFaceResult> results;
        int num_faces = face_detect_retian_run(ctx, resized, results, conf_thresh, nms_thresh, 10);
        auto detect_end = std::chrono::high_resolution_clock::now();
        total_detect_time += std::chrono::duration_cast<std::chrono::microseconds>(detect_end - detect_start).count();
        
        total_faces += num_faces;
        faces_per_frame.push_back(num_faces);
        
        // 只保留最近100帧的统计
        if (faces_per_frame.size() > 100) {
            faces_per_frame.erase(faces_per_frame.begin());
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
            
            // 计算平均检测人脸数
            float avg_faces = 0.0f;
            if (!faces_per_frame.empty()) {
                avg_faces = std::accumulate(faces_per_frame.begin(), faces_per_frame.end(), 0.0f) / faces_per_frame.size();
            }
            
            // 输出统计信息
            log_info("帧统计: 总帧数=%ld, FPS=%.1f, 检测人脸=%d(平均%.1f), 运行时间=%lds", 
                     total_frames, current_fps, num_faces, avg_faces, total_elapsed.count());
        }
        
        // 保存检测结果图片
        if (save_images && total_frames % save_interval == 0) {
            auto draw_start = std::chrono::high_resolution_clock::now();
            
            // 将检测结果映射回720p坐标（因为检测是在720p上做的）
            drawDetections(resized, results);
            
            auto draw_end = std::chrono::high_resolution_clock::now();
            total_draw_time += std::chrono::duration_cast<std::chrono::microseconds>(draw_end - draw_start).count();
            
            char filename[256];
            snprintf(filename, sizeof(filename), "retian_frame_%06ld_faces_%d.jpg", total_frames, num_faces);
            if (cv::imwrite(filename, resized)) {
                log_debug("已保存: %s", filename);
            }
        }
    }
    
    // 清理
    face_detect_retian_release(ctx);
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
    log_info("  总检测人脸数: %ld", total_faces);
    log_info("  平均每帧人脸数: %.2f", total_frames > 0 ? (double)total_faces / total_frames : 0.0);
    
    if (total_frames > 0) {
        log_info("性能分析:");
        log_info("  平均帧捕获耗时: %.2f ms", total_capture_time / 1000.0 / total_frames);
        log_info("  平均缩放耗时: %.2f ms", total_resize_time / 1000.0 / total_frames);
        log_info("  平均检测耗时: %.2f ms", total_detect_time / 1000.0 / total_frames);
        if (save_images && total_draw_time > 0) {
            log_info("  平均绘制耗时: %.2f ms", total_draw_time / 1000.0 / (total_frames / save_interval));
        }
    }
    
    log_info("=================================");
    
    return 0;
}
