#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

struct StreamOverlayTrack {
    int id{0};
    cv::Rect bbox;
    float personConfidence{0.0f};
    bool hasFaceBox{false};
    cv::Rect faceBox;
    float faceConfidence{0.0f};
    bool confirmed{false};
    bool approaching{false};
    bool hasCaptured{false};
    bool hasReversed{false};
    float trajectoryScore{0.0f};
    float peakBottom{0.0f};
    float peakArea{0.0f};
    std::vector<cv::Point> path;
};

struct StreamOverlayStats {
    long frameIndex{0};
    double fps{0.0};
    double envBrightness{0.0};
    int activeTracks{0};
};

void drawStreamOverlay(cv::Mat& frame,
                       const std::vector<StreamOverlayTrack>& tracks,
                       const StreamOverlayStats& stats);
