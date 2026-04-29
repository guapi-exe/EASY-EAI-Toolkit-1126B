#pragma once

#include "streaming/rtsp_streamer.h"
#include "streaming/stream_overlay.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <opencv2/core/mat.hpp>
#include <string>
#include <thread>
#include <vector>

struct RtspStreamOptions {
    int width{1280};
    int height{720};
    int fps{15};
    int bitrateKbps{2000};
    std::string port{"8554"};
    std::string mountPath{"/stream"};
};

class RtspStreamTask {
public:
    explicit RtspStreamTask(const RtspStreamOptions& options);
    ~RtspStreamTask();

    bool start();
    void stop();

    bool shouldAcceptFrame();
    void publishFrame(const cv::Mat& frame,
                      std::vector<StreamOverlayTrack> tracks,
                      const StreamOverlayStats& stats);

    bool isRunning() const;
    bool hasClient() const;
    std::string url() const;

private:
    static int64_t nowMs();
    void run();

    RtspStreamOptions options;
    RtspStreamerConfig streamerConfig;
    std::unique_ptr<RtspStreamer> streamer;

    std::atomic<bool> running{false};
    std::atomic<int64_t> lastAcceptedFrameMs{0};
    std::atomic<int64_t> minFrameIntervalMs{66};

    std::thread worker;
    std::mutex frameMutex;
    std::condition_variable frameCv;
    cv::Mat latestFrame;
    std::vector<StreamOverlayTrack> latestTracks;
    StreamOverlayStats latestStats;
    uint64_t latestSeq{0};
    uint64_t consumedSeq{0};
};
