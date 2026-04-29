#pragma once

#include <opencv2/core/mat.hpp>

#include <memory>
#include <string>

struct RtspStreamerConfig {
    int width{960};
    int height{540};
    int fps{10};
    int bitrateKbps{1000};
    std::string port{"8554"};
    std::string mountPath{"/stream"};
};

class RtspStreamer {
public:
    explicit RtspStreamer(const RtspStreamerConfig& config);
    ~RtspStreamer();

    bool start();
    void stop();
    bool pushFrame(const cv::Mat& frame);

    bool isRunning() const;
    bool hasClient() const;
    std::string url() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
