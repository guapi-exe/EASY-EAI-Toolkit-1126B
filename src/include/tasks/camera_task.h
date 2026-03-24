#pragma once
#include <opencv2/opencv.hpp>
#include "rknn_api.h"
#include "main.h"
#include <thread>
#include <atomic>
#include <functional>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <deque>

class CameraTask {
public:
    using UploadCallback = std::function<void(const cv::Mat& img, int id, const std::string& type)>;
    using PersonEventCallback = std::function<void(int personId, const std::string& eventType)>;

    CameraTask(const std::string& personModelPath,
               const std::string& faceModelPath,
               int cameraIndex = 0);

    ~CameraTask();

    void start();
    void stop();
    void setUploadCallback(UploadCallback cb);
    void setPersonEventCallback(PersonEventCallback cb);
    void setBrightnessBlackThreshold(double threshold) { brightnessBlackThreshold.store(threshold); }
    double getBrightnessBlackThreshold() const { return brightnessBlackThreshold.load(); }
    void captureSnapshot();
    double getEnvironmentBrightness() const { return environmentBrightness.load(); }
    
    // 帧数统计相关函数
    long getTotalFrames() const { return totalFrames; }
    double getCurrentFPS() const { return currentFPS; }
    void resetFrameCount() { totalFrames = 0; }

private:
    struct CandidateEvalJob {
        int trackId;
        cv::Mat personRoi;
        float areaRatio;
        float personOcclusion;
        float motionRatio;
    };

    void run();
    void captureLoop();
    void candidateEvalLoop(rknn_context faceCtx);
    bool enqueueCandidateEvaluation(CandidateEvalJob job);
    double computeFocusMeasure(const cv::Mat& img);
    bool isFrontalFace(const std::vector<cv::Point2f>& landmarks);
    bool isSideFace(const std::vector<cv::Point2f>& landmarks);
    void processFrame(const cv::Mat& frame, rknn_context personCtx);
    void updateFPS(); // 更新FPS计算
    
    std::string personModelPath;
    std::string faceModelPath;
    int cameraIndex;

    std::thread worker;
    std::thread captureWorker;
    std::thread candidateWorker;
    std::atomic<bool> running;
    std::atomic<bool> cameraOpened{false};
    UploadCallback uploadCallback;
    PersonEventCallback personEventCallback;

    std::mutex frameMutex;
    std::condition_variable frameCv;
    cv::Mat latestFrame;
    uint64_t latestFrameSeq{0};
    uint64_t consumedFrameSeq{0};

    std::mutex candidateEvalMutex;
    std::condition_variable candidateEvalCv;
    std::deque<CandidateEvalJob> candidateEvalQueue;
    std::unordered_map<int, int> pendingCandidateEvalByTrack;

    std::unordered_set<int> capturedPersonIds;
    std::unordered_set<int> capturedFaceIds;
    
    // 帧数统计相关成员变量
    std::atomic<long> totalFrames{0};
    std::atomic<double> currentFPS{0.0};
    std::chrono::steady_clock::time_point lastFPSUpdate;
    std::chrono::steady_clock::time_point startTime;
    long framesAtLastUpdate{0};
    
    // RGA硬件加速缓冲区
    unsigned char* resized_buffer_720p{nullptr};

    // 运动稳定性：记录每个track上一次中心点（720p坐标）
    std::unordered_map<int, cv::Point2f> lastTrackCenters;
    size_t candidateRoundRobinOffset{0};

    std::unordered_set<int> reportedPersonIds;
    std::atomic<double> environmentBrightness{0.0};
    std::atomic<double> brightnessBlackThreshold{CAMERA_BRIGHTNESS_BLACK_THRESHOLD};
    bool hadPersonsInScene{false};
};
