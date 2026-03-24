#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include "rknn_api.h"

class NAFNetTinyEnhancer {
public:
    explicit NAFNetTinyEnhancer(const std::string& modelPath);
    ~NAFNetTinyEnhancer();

    bool isReady();
    cv::Mat enhance(const cv::Mat& input);

private:
    bool init();
    void release();

    std::string modelPath;
    rknn_context ctx{0};
    bool initTried{false};
    bool ready{false};
    int inputWidth{256};
    int inputHeight{256};
    int inputChannels{3};
    rknn_tensor_format inputFmt{RKNN_TENSOR_NHWC};
    rknn_tensor_format outputFmt{RKNN_TENSOR_NHWC};
};