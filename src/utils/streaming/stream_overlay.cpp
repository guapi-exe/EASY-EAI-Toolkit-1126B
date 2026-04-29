#include "streaming/stream_overlay.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdio>

namespace {

cv::Scalar trackColor(const StreamOverlayTrack& track) {
    if (track.hasCaptured) {
        return cv::Scalar(128, 128, 128);
    }
    if (track.hasReversed) {
        return cv::Scalar(0, 128, 255);
    }
    if (track.approaching) {
        return cv::Scalar(0, 220, 0);
    }
    return cv::Scalar(255, 180, 0);
}

cv::Rect clampRect(const cv::Rect& rect, const cv::Size& size) {
    if (size.width <= 0 || size.height <= 0) {
        return cv::Rect();
    }
    int x = std::max(0, std::min(rect.x, size.width - 1));
    int y = std::max(0, std::min(rect.y, size.height - 1));
    int w = std::max(1, std::min(rect.width, size.width - x));
    int h = std::max(1, std::min(rect.height, size.height - y));
    return cv::Rect(x, y, w, h);
}

void drawLabel(cv::Mat& frame,
               const std::string& text,
               const cv::Point& origin,
               const cv::Scalar& color) {
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.48, 1, &baseline);
    cv::Rect bg(origin.x,
                std::max(0, origin.y - textSize.height - baseline - 4),
                textSize.width + 8,
                textSize.height + baseline + 6);
    bg = clampRect(bg, frame.size());
    cv::rectangle(frame, bg, cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(frame,
                text,
                cv::Point(bg.x + 4, bg.y + textSize.height + 1),
                cv::FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv::LINE_AA);
}

} // namespace

void drawStreamOverlay(cv::Mat& frame,
                       const std::vector<StreamOverlayTrack>& tracks,
                       const StreamOverlayStats& stats) {
    if (frame.empty()) {
        return;
    }

    char info[160];
    std::snprintf(info,
                  sizeof(info),
                  "frame=%ld fps=%.1f brightness=%.1f tracks=%d",
                  stats.frameIndex,
                  stats.fps,
                  stats.envBrightness,
                  stats.activeTracks);
    drawLabel(frame, info, cv::Point(10, 24), cv::Scalar(255, 255, 255));

    for (const auto& track : tracks) {
        cv::Rect box = clampRect(track.bbox, frame.size());
        if (box.empty()) {
            continue;
        }

        cv::Scalar color = trackColor(track);
        cv::rectangle(frame, box, color, 2, cv::LINE_AA);
        if (track.hasFaceBox) {
            cv::Rect faceBox = clampRect(track.faceBox, frame.size());
            if (!faceBox.empty()) {
                cv::Scalar faceColor(0, 255, 255);
                cv::rectangle(frame, faceBox, faceColor, 2, cv::LINE_AA);
                char faceLabel[64];
                std::snprintf(faceLabel,
                              sizeof(faceLabel),
                              "face %.2f",
                              track.faceConfidence);
                drawLabel(frame, faceLabel, cv::Point(faceBox.x, std::max(18, faceBox.y - 4)), faceColor);
            }
        }

        if (track.path.size() >= 2) {
            std::vector<cv::Point> safePath;
            safePath.reserve(track.path.size());
            for (const auto& p : track.path) {
                safePath.emplace_back(
                    std::max(0, std::min(p.x, frame.cols - 1)),
                    std::max(0, std::min(p.y, frame.rows - 1)));
            }
            cv::polylines(frame, safePath, false, color, 2, cv::LINE_AA);
            cv::circle(frame, safePath.back(), 4, color, cv::FILLED, cv::LINE_AA);
        }

        char label[128];
        std::snprintf(label,
                      sizeof(label),
                      "ID %d person=%.2f %s score=%.2f peak=%.2f",
                      track.id,
                      track.personConfidence,
                      track.approaching ? "approach" : "track",
                      track.trajectoryScore,
                      track.peakBottom);
        drawLabel(frame, label, cv::Point(box.x, std::max(18, box.y - 4)), color);
    }
}
