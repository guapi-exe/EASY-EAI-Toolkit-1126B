#pragma once

#include "camera_task.h"
#include "person_detect.h"
#include "face_detect.h"
#include "sort_tracker.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <dirent.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>

extern "C" {
#include "log.h"
#include "camera.h"
#include "rga_wrapper.h"
}

namespace camera_task_internal {

enum class IrCutMode {
    Unknown = 0,
    White,
    Black,
};

struct AdaptiveCaptureThresholds {
    float lowLightStrength{0.0f};
    double minClarity{CAPTURE_MIN_CLARITY};
    double fallbackMinClarity{CAPTURE_FALLBACK_MIN_CLARITY};
    float maxMotionRatio{CAPTURE_MAX_MOTION_RATIO};
    float maxMotionRejectRatio{CAPTURE_MAX_MOTION_REJECT_RATIO};
    float maxBlurSeverity{CAPTURE_MAX_BLUR_SEVERITY};
    float fallbackMaxBlurSeverity{CAPTURE_FALLBACK_MAX_BLUR_SEVERITY};
};

struct MultiFrameFusionResult {
    cv::Mat fused;
    int acceptedFrames{0};
    float meanSimilarity{0.0f};
    double referenceFocus{0.0};
    double bestAlignedFocus{0.0};
};

constexpr int kIrCutMinSwitchIntervalSec = 60;
constexpr int kIrCutConsecutiveHits = 3;
constexpr int kIrCutSettleAfterSwitchSec = 8;
constexpr int kRejectLogThrottleMs = 1500;

constexpr double kLowLightBrightnessThreshold = 92.0;
constexpr double kLowLightBrightnessFloor = 58.0;
constexpr float kLowLightMotionRatioScale = 0.72f;
constexpr float kLowLightMotionRejectRatioScale = 0.60f;
constexpr float kLowLightMaxBlurSeverityScale = 0.82f;
constexpr float kLowLightFallbackMaxBlurSeverityScale = 0.74f;
constexpr float kLowLightMinClarityScale = 1.10f;
constexpr float kLowLightFallbackMinClarityScale = 1.18f;
constexpr size_t kMultiFrameFusionHistorySize = 3;
constexpr float kMultiFrameFusionLowLightMinStrength = 0.18f;
constexpr float kMultiFrameFusionMaxShiftRatio = 0.20f;
constexpr float kMultiFrameFusionMinSimilarity = 0.36f;
constexpr int kApproachPositiveFramesRequired = 2;
constexpr int kApproachNegativeFramesRequired = 3;
constexpr float kApproachJitterFreezeThreshold = 0.16f;
constexpr float kApproachJitterRejectThreshold = 0.30f;

extern IrCutMode g_irCutMode;
extern bool g_irCutGpioReady;
extern std::chrono::steady_clock::time_point g_lastIrCutSwitchTime;

float clampUnit(float value);
float lerpFloat(float a, float b, float t);
double lerpDouble(double a, double b, float t);

AdaptiveCaptureThresholds buildAdaptiveCaptureThresholds(
    const DeviceConfig::CaptureDefaults& config,
    double sceneBrightness);

double updateFilteredBrightness(double prev, double raw);
const char* irCutModeToString(IrCutMode mode);
bool canSwitchIrCutNow();
bool writeSysfsValue(const std::string& path, const std::string& value);
void ensureIrCutGpioReady();
void switchIrCutWhite();
void switchIrCutBlack();

double computeSceneBrightnessFast(const cv::Mat& frame);
bool applyBrightnessBoost(
    cv::Mat& frame,
    double currentBrightness,
    const DeviceConfig::BrightnessBoostConfig& config);

bool isValidTrackRect(const cv::Rect2f& rect);
cv::Rect2f selectTrackRect720p(const Track& track);
float computeTrackAreaTrendRatio(const Track& track);
cv::Rect clampRectToSize(const cv::Rect& rect, const cv::Size& bounds);
cv::Rect expandRectFromCenter(const cv::Rect& rect, float scale, const cv::Size& bounds);
double computeGrayFocusVariance(const cv::Mat& gray);

MultiFrameFusionResult fuseTrackHistoryPersonRoi(
    const cv::Mat& reference,
    const std::vector<cv::Mat>& history,
    const cv::Rect& focusBox,
    float lowLightStrength,
    float motionRatio);

float rect_iou(const cv::Rect& a, const cv::Rect& b);
float rect_overlap_ratio_on_a(const cv::Rect& a, const cv::Rect& b);
void nmsDetections(std::vector<Detection>& dets, float iouThreshold);

}  // namespace camera_task_internal
