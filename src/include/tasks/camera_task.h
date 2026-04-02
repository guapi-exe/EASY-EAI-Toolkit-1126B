#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include "device_config.h"
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
#include <vector>

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
    void setRuntimeConfig(const DeviceConfig& config);
    void setBrightnessBlackThreshold(double threshold) { brightnessBlackThreshold.store(threshold); }
    double getBrightnessBlackThreshold() const { return brightnessBlackThreshold.load(); }
    void captureSnapshot();
    double getEnvironmentBrightness() const { return environmentBrightness.load(); }

    long getTotalFrames() const { return totalFrames; }
    double getCurrentFPS() const { return currentFPS; }
    void resetFrameCount() { totalFrames = 0; }

private:
    struct CandidateEvalJob {
        int trackId;
        cv::Mat personRoi;
        std::vector<cv::Mat> fusionHistory;
        float areaRatio;
        float personOcclusion;
        float motionRatio;
    };

    struct TrackApproachState {
        bool isApproaching{false};
        int positiveHits{0};
        int negativeHits{0};
        float lastTrend{0.0f};
        float lastJitter{0.0f};
        float lastAreaRatio{0.0f};
        size_t lastHistorySize{0};
    };

    void run();
    void captureLoop();
    void candidateEvalLoop(rknn_context faceCtx);
    bool enqueueCandidateEvaluation(CandidateEvalJob job);
    double computeFocusMeasure(const cv::Mat& img);
    float computeMotionBlurSeverity(const cv::Mat& img);
    bool isFrontalFace(const std::vector<cv::Point2f>& landmarks);
    bool isSideFace(const std::vector<cv::Point2f>& landmarks);
    void logTrackReject(const char* stage, int trackId, const char* reason, const std::string& detail);
    void clearTrackReject(const char* stage, int trackId);
    void processFrame(const cv::Mat& frame, rknn_context personCtx);
    void updateFPS();
    DeviceConfig::CaptureDefaults getCaptureConfigSnapshot() const;
    DeviceConfig::BrightnessBoostConfig getBrightnessBoostConfigSnapshot() const;

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

    struct RejectLogState {
        std::string reason;
        int suppressedCount{0};
        std::chrono::steady_clock::time_point lastLogTime{};
    };
    std::mutex rejectLogMutex;
    std::unordered_map<std::string, RejectLogState> rejectLogStates;

    std::unordered_set<int> capturedPersonIds;
    std::unordered_set<int> capturedFaceIds;

    std::atomic<long> totalFrames{0};
    std::atomic<double> currentFPS{0.0};
    std::chrono::steady_clock::time_point lastFPSUpdate;
    std::chrono::steady_clock::time_point startTime;
    long framesAtLastUpdate{0};

    std::unordered_map<int, cv::Point2f> lastTrackCenters;
    std::unordered_map<int, std::deque<cv::Mat>> trackPersonRoiHistory;
    std::unordered_map<int, TrackApproachState> trackApproachStates;
    size_t candidateRoundRobinOffset{0};

    std::unordered_set<int> reportedPersonIds;
    std::atomic<double> environmentBrightness{0.0};
    std::atomic<double> brightnessBlackThreshold{CAMERA_BRIGHTNESS_BLACK_THRESHOLD};
    mutable std::mutex configMutex;
    DeviceConfig::CaptureDefaults captureConfig;
    DeviceConfig::BrightnessBoostConfig brightnessBoostConfig;
    bool hadPersonsInScene{false};
};
