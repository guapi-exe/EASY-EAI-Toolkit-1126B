#pragma once
#include <opencv2/opencv.hpp>
#include "rknn_api.h"
#include <thread>
#include <atomic>
#include <functional>
#include <unordered_set>
#include <string>

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

private:
    void run();
    double computeFocusMeasure(const cv::Mat& img);
    void processFrame(const cv::Mat& frame, rknn_context personCtx, rknn_context faceCtx);
    std::string personModelPath;
    std::string faceModelPath;
    int cameraIndex;

    std::thread worker;
    std::atomic<bool> running;
    UploadCallback uploadCallback;

    std::unordered_set<int> capturedPersonIds;
    std::unordered_set<int> capturedFaceIds;
};
