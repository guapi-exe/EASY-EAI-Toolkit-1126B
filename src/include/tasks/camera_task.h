#pragma once
#include <opencv2/opencv.hpp>
#include "rknn_api.h"
#include <thread>
#include <atomic>
#include <functional>
#include <unordered_set>
#include <string>
#include <chrono>

class CameraTask {
public:
    using UploadCallback = std::function<void(const cv::Mat& img, int id, const std::string& type)>;

    CameraTask(const std::string& personModelPath,
               const std::string& faceModelPath,
               int cameraIndex = 0);

    ~CameraTask();

    void start();
    void stop();
    void setUploadCallback(UploadCallback cb);
    void captureSnapshot();
    
    // 帧数统计相关函数
    long getTotalFrames() const { return totalFrames; }
    double getCurrentFPS() const { return currentFPS; }
    void resetFrameCount() { totalFrames = 0; }

private:
    void run();
    double computeFocusMeasure(const cv::Mat& img);
    void processFrame(const cv::Mat& frame, rknn_context personCtx, rknn_context faceCtx);
    void updateFPS(); // 更新FPS计算
    
    std::string personModelPath;
    std::string faceModelPath;
    int cameraIndex;

    std::thread worker;
    std::atomic<bool> running;
    UploadCallback uploadCallback;

    std::unordered_set<int> capturedPersonIds;
    std::unordered_set<int> capturedFaceIds;
    
    // 帧数统计相关成员变量
    std::atomic<long> totalFrames{0};
    std::atomic<double> currentFPS{0.0};
    std::chrono::steady_clock::time_point lastFPSUpdate;
    std::chrono::steady_clock::time_point startTime;
    long framesAtLastUpdate{0};
};
