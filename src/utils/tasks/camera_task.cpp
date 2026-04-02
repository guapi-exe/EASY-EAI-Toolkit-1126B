#include "camera_task.h"
#include "main.h"
#include "person_detect.h"
#include "face_detect.h"
#include "sort_tracker.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <dirent.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>
extern "C" {
#include "log.h"
#include "camera.h"
#include "rga_wrapper.h"
}

using namespace cv;
using namespace std;

namespace {
enum class IrCutMode {
    Unknown = 0,
    White,
    Black,
};

static IrCutMode g_irCutMode = IrCutMode::Unknown;
static bool g_irCutGpioReady = false;
static std::chrono::steady_clock::time_point g_lastIrCutSwitchTime =
    std::chrono::steady_clock::now() - std::chrono::seconds(3600);

constexpr int kIrCutMinSwitchIntervalSec = 60;
constexpr int kIrCutConsecutiveHits = 3;
constexpr int kIrCutSettleAfterSwitchSec = 8;
constexpr int kRejectLogThrottleMs = 1500;

struct AdaptiveCaptureThresholds {
    float lowLightStrength{0.0f};
    double minClarity{CAPTURE_MIN_CLARITY};
    double fallbackMinClarity{CAPTURE_FALLBACK_MIN_CLARITY};
    float maxMotionRatio{CAPTURE_MAX_MOTION_RATIO};
    float maxMotionRejectRatio{CAPTURE_MAX_MOTION_REJECT_RATIO};
    float maxBlurSeverity{CAPTURE_MAX_BLUR_SEVERITY};
    float fallbackMaxBlurSeverity{CAPTURE_FALLBACK_MAX_BLUR_SEVERITY};
};

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

struct MultiFrameFusionResult {
    cv::Mat fused;
    int acceptedFrames{0};
    float meanSimilarity{0.0f};
    double referenceFocus{0.0};
    double bestAlignedFocus{0.0};
};

float clampUnit(float value) {
    return std::max(0.0f, std::min(1.0f, value));
}

float lerpFloat(float a, float b, float t) {
    return a + (b - a) * t;
}

double lerpDouble(double a, double b, float t) {
    return a + (b - a) * static_cast<double>(t);
}

AdaptiveCaptureThresholds buildAdaptiveCaptureThresholds(const DeviceConfig::CaptureDefaults& config,
                                                         double sceneBrightness) {
    AdaptiveCaptureThresholds thresholds;
    thresholds.minClarity = config.minClarity;
    thresholds.fallbackMinClarity = config.fallbackMinClarity;
    thresholds.maxMotionRatio = config.maxMotionRatio;
    thresholds.maxMotionRejectRatio = config.maxMotionRejectRatio;
    thresholds.maxBlurSeverity = config.maxBlurSeverity;
    thresholds.fallbackMaxBlurSeverity = config.fallbackMaxBlurSeverity;

    if (sceneBrightness <= 1.0) {
        return thresholds;
    }

    double low_light_start = std::max(kLowLightBrightnessThreshold,
                                      kLowLightBrightnessFloor + 1.0);
    double low_light_floor = std::min(kLowLightBrightnessFloor,
                                      low_light_start - 1.0);
    double range = std::max(8.0, low_light_start - low_light_floor);

    float low_light_strength = 0.0f;
    if (sceneBrightness < low_light_start) {
        low_light_strength = clampUnit(static_cast<float>((low_light_start - sceneBrightness) / range));
    }

    thresholds.lowLightStrength = low_light_strength;
    thresholds.minClarity = lerpDouble(config.minClarity,
                                       config.minClarity * kLowLightMinClarityScale,
                                       low_light_strength);
    thresholds.fallbackMinClarity = lerpDouble(config.fallbackMinClarity,
                                               config.fallbackMinClarity * kLowLightFallbackMinClarityScale,
                                               low_light_strength);
    thresholds.maxMotionRatio = std::max(0.0015f,
                                         lerpFloat(config.maxMotionRatio,
                                                   config.maxMotionRatio * kLowLightMotionRatioScale,
                                                   low_light_strength));
    thresholds.maxMotionRejectRatio = std::max(thresholds.maxMotionRatio + 0.003f,
                                               lerpFloat(config.maxMotionRejectRatio,
                                                         config.maxMotionRejectRatio * kLowLightMotionRejectRatioScale,
                                                         low_light_strength));
    thresholds.maxBlurSeverity = std::max(0.12f,
                                          lerpFloat(config.maxBlurSeverity,
                                                    config.maxBlurSeverity * kLowLightMaxBlurSeverityScale,
                                                    low_light_strength));
    thresholds.fallbackMaxBlurSeverity = std::max(thresholds.maxBlurSeverity + 0.04f,
                                                  lerpFloat(config.fallbackMaxBlurSeverity,
                                                            config.fallbackMaxBlurSeverity * kLowLightFallbackMaxBlurSeverityScale,
                                                            low_light_strength));
    return thresholds;
}

double updateFilteredBrightness(double prev, double raw) {
    if (prev < 0.0) {
        return raw;
    }

    double diff = std::fabs(raw - prev);
    double alpha = 0.20;
    if (diff >= 40.0) {
        alpha = 0.75;
    } else if (diff >= 20.0) {
        alpha = 0.55;
    } else if (diff >= 8.0) {
        alpha = 0.35;
    }

    return prev * (1.0 - alpha) + raw * alpha;
}

const char* irCutModeToString(IrCutMode mode) {
    switch (mode) {
        case IrCutMode::White:
            return "WHITE";
        case IrCutMode::Black:
            return "BLACK";
        default:
            return "UNKNOWN";
    }
}

bool canSwitchIrCutNow() {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - g_lastIrCutSwitchTime).count();
    return elapsed >= kIrCutMinSwitchIntervalSec;
}

bool writeSysfsValue(const std::string& path, const std::string& value) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        return false;
    }
    ofs << value;
    return ofs.good();
}

void ensureIrCutGpioReady() {
    if (g_irCutGpioReady) {
        return;
    }

    // 重复export允许失败(已经export会返回busy)
    writeSysfsValue("/sys/class/gpio/export", "184");
    writeSysfsValue("/sys/class/gpio/export", "185");
    writeSysfsValue("/sys/class/gpio/gpio184/direction", "out");
    writeSysfsValue("/sys/class/gpio/gpio185/direction", "out");

    g_irCutGpioReady = true;
}

void switchIrCutWhite() {
    if (g_irCutMode == IrCutMode::White) {
        return;
    }
    if (!canSwitchIrCutNow()) {
        return;
    }
    ensureIrCutGpioReady();
    writeSysfsValue("/sys/class/gpio/gpio184/value", "1");
    writeSysfsValue("/sys/class/gpio/gpio185/value", "0");
    writeSysfsValue("/sys/class/gpio/gpio184/value", "0");
    g_irCutMode = IrCutMode::White;
    g_lastIrCutSwitchTime = std::chrono::steady_clock::now();
    log_info("CameraTask: IR-CUT switched to WHITE mode");
}

void switchIrCutBlack() {
    if (g_irCutMode == IrCutMode::Black) {
        return;
    }
    if (!canSwitchIrCutNow()) {
        return;
    }
    ensureIrCutGpioReady();
    writeSysfsValue("/sys/class/gpio/gpio184/value", "0");
    writeSysfsValue("/sys/class/gpio/gpio185/value", "1");
    writeSysfsValue("/sys/class/gpio/gpio184/value", "1");
    g_irCutMode = IrCutMode::Black;
    g_lastIrCutSwitchTime = std::chrono::steady_clock::now();
    log_info("CameraTask: IR-CUT switched to BLACK mode");
}

double computeSceneBrightnessFast(const cv::Mat& frame) {
    if (frame.empty()) {
        return 0.0;
    }

    // 低频调用(每N帧一次)下使用缩小+灰度均值更稳，避免手写采样在不同像素格式下偏差。
    cv::Mat small;
    cv::resize(frame, small, cv::Size(64, 36), 0, 0, cv::INTER_AREA);

    cv::Mat gray;
    if (small.channels() == 3) {
        cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
    } else if (small.channels() == 1) {
        gray = small;
    } else {
        return 0.0;
    }

    return cv::mean(gray)[0];
}

// ─── 软件亮度补偿 ──────────────────────────────────────────────
// 当 ISP 输出偏暗（filteredBrightness < CAMERA_BRIGHTNESS_BOOST_THRESHOLD）时，
// 用 Gamma + linear gain 将画面提亮。Gamma 比纯线性更好：暗部拉得多、亮部不过曝。
// 每帧只在需要时执行，性能开销约 3-4 ms（2688x1520 BGR, RK3568 A55）。
static cv::Mat g_gammaLUT;
static double  g_gammaLUT_gamma = -1.0;

static void buildGammaLUT(double gamma) {
    if (std::fabs(gamma - g_gammaLUT_gamma) < 1e-6 && !g_gammaLUT.empty()) {
        return; // 已缓存
    }
    cv::Mat lut(1, 256, CV_8UC1);
    uchar* p = lut.ptr();
    for (int i = 0; i < 256; i++) {
        p[i] = cv::saturate_cast<uchar>(std::pow(i / 255.0, gamma) * 255.0);
    }
    g_gammaLUT = lut;
    g_gammaLUT_gamma = gamma;
}

// 返回 true 表示实际做了提亮
static bool applyBrightnessBoost(cv::Mat& frame,
                                 double currentBrightness,
                                 const DeviceConfig::BrightnessBoostConfig& config) {
    if (currentBrightness >= config.boostThreshold
        || currentBrightness < config.boostMinFloor) {
        return false;
    }

    // ── 自适应目标：极暗场景不硬拉到满目标，限制噪声放大 ──
    double effectiveTarget = config.target;
    if (currentBrightness < 55.0) {
        // 亮度很暗时只拉向部分目标，避免噪声被过度放大。
        effectiveTarget = currentBrightness
            + (config.target - currentBrightness) * config.darkBlend;
    }

    // Gamma 校正：对暗部提升更温和，保留高光
    double gamma = config.gamma;
    if (currentBrightness < 50.0) {
        gamma = std::max(0.60, gamma - 0.10);
    }
    buildGammaLUT(gamma);
    // cv::LUT 对多通道 Mat 自动逐通道应用单通道 LUT
    cv::LUT(frame, g_gammaLUT, frame);

    // Gamma 后测量残差，只在仍偏暗时施加轻量线性补偿
    double afterGamma = cv::mean(frame)[0];
    if (afterGamma < effectiveTarget && afterGamma > 1.0) {
        double residualAlpha = std::min(effectiveTarget / afterGamma, config.maxAlpha);
        double residualBeta  = std::min((effectiveTarget - afterGamma) * 0.10, config.maxBeta);
        if (residualAlpha > 1.03 || residualBeta > 1.5) {
            frame.convertTo(frame, -1, residualAlpha, residualBeta);
        }
    }
    return true;
}

bool isValidTrackRect(const cv::Rect2f& rect) {
    return rect.width > 1.0f && rect.height > 1.0f;
}

cv::Rect2f selectTrackRect720p(const Track& track) {
    return isValidTrackRect(track.smoothed_bbox) ? track.smoothed_bbox : track.bbox;
}

float computeTrackAreaTrendRatio(const Track& track) {
    if (track.bbox_history.size() < 4) {
        return 0.0f;
    }

    size_t span = std::min<size_t>(4, track.bbox_history.size() - 1);
    float area_now = track.bbox_history.back();
    float area_prev = track.bbox_history[track.bbox_history.size() - 1 - span];
    return (area_now - area_prev) / (area_prev + 1e-6f);
}

cv::Rect clampRectToSize(const cv::Rect& rect, const cv::Size& bounds) {
    int x = std::max(0, std::min(rect.x, bounds.width - 1));
    int y = std::max(0, std::min(rect.y, bounds.height - 1));
    int width = std::max(1, std::min(rect.width, bounds.width - x));
    int height = std::max(1, std::min(rect.height, bounds.height - y));
    return cv::Rect(x, y, width, height);
}

cv::Rect expandRectFromCenter(const cv::Rect& rect, float scale, const cv::Size& bounds) {
    float cx = rect.x + rect.width * 0.5f;
    float cy = rect.y + rect.height * 0.5f;
    float width = rect.width * scale;
    float height = rect.height * scale;
    cv::Rect expanded(static_cast<int>(std::round(cx - width * 0.5f)),
                      static_cast<int>(std::round(cy - height * 0.5f)),
                      static_cast<int>(std::round(width)),
                      static_cast<int>(std::round(height)));
    return clampRectToSize(expanded, bounds);
}

double computeGrayFocusVariance(const cv::Mat& gray) {
    if (gray.empty() || gray.cols <= 1 || gray.rows <= 1) {
        return 0.0;
    }

    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_32F);
    cv::Scalar mean_val, stddev_val;
    cv::meanStdDev(lap, mean_val, stddev_val);
    return stddev_val[0] * stddev_val[0];
}

MultiFrameFusionResult fuseTrackHistoryPersonRoi(const cv::Mat& reference,
                                                 const std::vector<cv::Mat>& history,
                                                 const cv::Rect& focus_box,
                                                 float low_light_strength,
                                                 float motion_ratio) {
    MultiFrameFusionResult result;
    if (reference.empty() || history.empty() || focus_box.width < 16 || focus_box.height < 16) {
        return result;
    }

    cv::Rect safe_focus = expandRectFromCenter(focus_box, 1.45f, reference.size());
    cv::Mat reference_gray;
    cv::cvtColor(reference, reference_gray, cv::COLOR_BGR2GRAY);
    result.referenceFocus = computeGrayFocusVariance(reference_gray(safe_focus));
    result.bestAlignedFocus = result.referenceFocus;

    cv::Mat reference_focus_f;
    reference_gray(safe_focus).convertTo(reference_focus_f, CV_32F);
    cv::GaussianBlur(reference_focus_f, reference_focus_f, cv::Size(0, 0), 0.8);

    float blur_sigma = 1.10f + low_light_strength * 0.90f;
    cv::Mat reference_base;
    cv::GaussianBlur(reference, reference_base, cv::Size(0, 0), blur_sigma);

    cv::Mat accum;
    reference_base.convertTo(accum, CV_32FC3);
    accum *= 1.28f;
    cv::Mat weight_sum(reference.size(), CV_32F, cv::Scalar(1.28f));

    cv::Mat sharpest = reference.clone();
    float similarity_sum = 0.0f;
    int similarity_count = 0;

    size_t history_begin = history.size() > kMultiFrameFusionHistorySize ?
        history.size() - kMultiFrameFusionHistorySize : 0;
    for (size_t i = history_begin; i < history.size(); ++i) {
        if (history[i].empty()) {
            continue;
        }

        cv::Mat candidate = history[i];
        if (candidate.size() != reference.size()) {
            cv::resize(candidate, candidate, reference.size(), 0, 0, cv::INTER_LINEAR);
        }

        cv::Mat candidate_gray;
        cv::cvtColor(candidate, candidate_gray, cv::COLOR_BGR2GRAY);
        cv::Mat candidate_focus_f;
        candidate_gray(safe_focus).convertTo(candidate_focus_f, CV_32F);
        cv::GaussianBlur(candidate_focus_f, candidate_focus_f, cv::Size(0, 0), 0.8);

        cv::Point2d shift = cv::phaseCorrelate(reference_focus_f, candidate_focus_f);
        float shift_norm = std::sqrt(static_cast<float>(shift.x * shift.x + shift.y * shift.y));
        float max_shift = std::max(4.0f, std::min(safe_focus.width, safe_focus.height) * kMultiFrameFusionMaxShiftRatio);
        if (shift_norm > max_shift) {
            continue;
        }

        cv::Mat transform = (cv::Mat_<double>(2, 3) << 1.0, 0.0, shift.x,
                                                       0.0, 1.0, shift.y);
        cv::Mat aligned;
        cv::warpAffine(candidate, aligned, transform, reference.size(),
                       cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        cv::Mat aligned_gray;
        cv::cvtColor(aligned, aligned_gray, cv::COLOR_BGR2GRAY);
        cv::Mat diff;
        cv::absdiff(aligned_gray, reference_gray, diff);
        cv::Mat diff_f;
        diff.convertTo(diff_f, CV_32F);
        cv::Mat similarity = 1.0f - (diff_f - 14.0f) / 62.0f;
        cv::max(similarity, 0.0f, similarity);
        cv::min(similarity, 1.0f, similarity);
        cv::GaussianBlur(similarity, similarity, cv::Size(0, 0), 1.2);

        float mean_similarity = static_cast<float>(cv::mean(similarity(safe_focus))[0]);
        if (mean_similarity < kMultiFrameFusionMinSimilarity) {
            continue;
        }

        cv::Mat aligned_base;
        cv::GaussianBlur(aligned, aligned_base, cv::Size(0, 0), blur_sigma);
        cv::Mat aligned_base_f;
        aligned_base.convertTo(aligned_base_f, CV_32FC3);

        float frame_weight_scale = 0.40f + low_light_strength * 0.18f;
        if (motion_ratio > 0.010f) {
            frame_weight_scale *= 0.92f;
        }
        cv::Mat frame_weight = similarity * frame_weight_scale;
        std::vector<cv::Mat> weight_channels(3, frame_weight);
        cv::Mat frame_weight3;
        cv::merge(weight_channels, frame_weight3);
        accum += aligned_base_f.mul(frame_weight3);
        weight_sum += frame_weight;

        double aligned_focus = computeGrayFocusVariance(aligned_gray(safe_focus));
        if (aligned_focus > result.bestAlignedFocus * 1.02) {
            sharpest = aligned;
            result.bestAlignedFocus = aligned_focus;
        }

        similarity_sum += mean_similarity;
        similarity_count++;
        result.acceptedFrames++;
    }

    if (result.acceptedFrames <= 0) {
        result.acceptedFrames = 1;
        result.meanSimilarity = 0.0f;
        return result;
    }

    result.meanSimilarity = similarity_count > 0 ? (similarity_sum / similarity_count) : 0.0f;

    cv::Mat safe_weight = weight_sum + 1e-4f;
    std::vector<cv::Mat> safe_weight_channels(3, safe_weight);
    cv::Mat safe_weight3;
    cv::merge(safe_weight_channels, safe_weight3);
    cv::Mat fused_base = accum / safe_weight3;

    cv::Mat sharpest_base;
    cv::GaussianBlur(sharpest, sharpest_base, cv::Size(0, 0), blur_sigma);
    cv::Mat sharpest_f;
    sharpest.convertTo(sharpest_f, CV_32FC3);
    cv::Mat sharpest_base_f;
    sharpest_base.convertTo(sharpest_base_f, CV_32FC3);
    cv::Mat detail = sharpest_f - sharpest_base_f;

    cv::Mat reference_f;
    reference.convertTo(reference_f, CV_32FC3);
    float detail_gain = 0.78f + low_light_strength * 0.14f;
    float reference_mix = 0.16f + std::max(0.0f, motion_ratio - 0.006f) * 10.0f;
    reference_mix = std::max(0.16f, std::min(0.34f, reference_mix));

    cv::Mat fused_f = fused_base + detail * detail_gain;
    fused_f = fused_f * (1.0f - reference_mix) + reference_f * reference_mix;
    fused_f.convertTo(result.fused, CV_8UC3);
    return result;
}
} // namespace

static float rect_iou(const cv::Rect& a, const cv::Rect& b) {
    int xx1 = std::max(a.x, b.x);
    int yy1 = std::max(a.y, b.y);
    int xx2 = std::min(a.x + a.width, b.x + b.width);
    int yy2 = std::min(a.y + a.height, b.y + b.height);
    int w = std::max(0, xx2 - xx1);
    int h = std::max(0, yy2 - yy1);
    float inter = static_cast<float>(w * h);
    float uni = static_cast<float>(a.area() + b.area()) - inter;
    return inter / (uni + 1e-6f);
}

static float rect_overlap_ratio_on_a(const cv::Rect& a, const cv::Rect& b) {
    int xx1 = std::max(a.x, b.x);
    int yy1 = std::max(a.y, b.y);
    int xx2 = std::min(a.x + a.width, b.x + b.width);
    int yy2 = std::min(a.y + a.height, b.y + b.height);
    int w = std::max(0, xx2 - xx1);
    int h = std::max(0, yy2 - yy1);
    float inter = static_cast<float>(w * h);
    float a_area = static_cast<float>(std::max(1, a.area()));
    return inter / a_area;
}

static void nmsDetections(std::vector<Detection>& dets, float iouThreshold) {
    if (dets.size() <= 1) {
        return;
    }

    std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) {
        return a.prop > b.prop;
    });

    std::vector<bool> suppressed(dets.size(), false);
    std::vector<Detection> kept;
    kept.reserve(dets.size());

    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }

        kept.push_back(dets[i]);
        cv::Rect a(static_cast<int>(dets[i].x1), static_cast<int>(dets[i].y1),
                   static_cast<int>(dets[i].x2 - dets[i].x1),
                   static_cast<int>(dets[i].y2 - dets[i].y1));

        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j]) {
                continue;
            }
            cv::Rect b(static_cast<int>(dets[j].x1), static_cast<int>(dets[j].y1),
                       static_cast<int>(dets[j].x2 - dets[j].x1),
                       static_cast<int>(dets[j].y2 - dets[j].y1));
            if (rect_iou(a, b) > iouThreshold) {
                suppressed[j] = true;
            }
        }
    }

    dets.swap(kept);
}

CameraTask::CameraTask(const string& personModel, const string& faceModel, int index)
    : personModelPath(personModel), faceModelPath(faceModel), cameraIndex(index), running(false) {
    startTime = std::chrono::steady_clock::now();
    lastFPSUpdate = startTime;
    
    rga_init();
    resized_buffer_720p = new unsigned char[IMAGE_WIDTH * IMAGE_HEIGHT * 3];
}

CameraTask::~CameraTask() { 
    stop(); 
    
    if (resized_buffer_720p) {
        delete[] resized_buffer_720p;
        resized_buffer_720p = nullptr;
    }
    rga_unInit();
}

void CameraTask::start() {
    if (running) {
        //log_warn("CameraTask: already running, ignoring start request");
        return;
    }
    running = true;
    log_info("CameraTask: launching worker thread...");
    worker = thread(&CameraTask::run, this);
}

void CameraTask::stop() {
    if (!running && !cameraOpened && !worker.joinable()) return;
    log_info("CameraTask: stopping...");
    running = false;
    frameCv.notify_all();
    {
        std::lock_guard<std::mutex> lock(candidateEvalMutex);
        candidateEvalQueue.clear();
        pendingCandidateEvalByTrack.clear();
    }
    candidateEvalCv.notify_all();

    // 主动关闭摄像头，打断可能阻塞的取帧调用
    if (cameraOpened.exchange(false)) {
        mipicamera_exit(cameraIndex);
    }

    if (worker.joinable()) worker.join();
    trackPersonRoiHistory.clear();
    log_info("CameraTask: stopped");
}

void CameraTask::setUploadCallback(UploadCallback cb) {
    uploadCallback = cb;
}

void CameraTask::setPersonEventCallback(PersonEventCallback cb) {
    personEventCallback = cb;
}

void CameraTask::setRuntimeConfig(const DeviceConfig& config) {
    {
        std::lock_guard<std::mutex> lock(configMutex);
        captureConfig = config.captureDefaults;
        brightnessBoostConfig = config.brightnessBoost;
    }

    brightnessBlackThreshold.store(config.captureDefaults.brightnessBlackThreshold);
    set_max_frame_candidates(static_cast<size_t>(std::max(1, config.captureDefaults.maxFrameCandidates)));
}

DeviceConfig::CaptureDefaults CameraTask::getCaptureConfigSnapshot() const {
    std::lock_guard<std::mutex> lock(configMutex);
    return captureConfig;
}

DeviceConfig::BrightnessBoostConfig CameraTask::getBrightnessBoostConfigSnapshot() const {
    std::lock_guard<std::mutex> lock(configMutex);
    return brightnessBoostConfig;
}

void CameraTask::logTrackReject(const char* stage, int trackId, const char* reason, const std::string& detail) {
    auto now = std::chrono::steady_clock::now();
    std::string key = std::string(stage) + ":" + std::to_string(trackId);

    std::lock_guard<std::mutex> lock(rejectLogMutex);
    auto& state = rejectLogStates[key];

    bool has_prev_log = state.lastLogTime.time_since_epoch().count() != 0;
    bool same_reason = state.reason == reason;
    bool within_throttle = has_prev_log &&
        std::chrono::duration_cast<std::chrono::milliseconds>(now - state.lastLogTime).count() < kRejectLogThrottleMs;

    if (same_reason && within_throttle) {
        state.suppressedCount++;
        return;
    }

    char suffix[64] = {0};
    if (state.suppressedCount > 0) {
        std::snprintf(suffix, sizeof(suffix), " suppressed=%d", state.suppressedCount);
    }

    state.reason = reason;
    state.suppressedCount = 0;
    state.lastLogTime = now;

    log_debug("Track %d %s reject[%s]: %s%s", trackId, stage, reason, detail.c_str(), suffix);
}

void CameraTask::clearTrackReject(const char* stage, int trackId) {
    std::string key = std::string(stage) + ":" + std::to_string(trackId);
    std::lock_guard<std::mutex> lock(rejectLogMutex);
    rejectLogStates.erase(key);
}

// -------------------- 图像清晰度计算 --------------------
double CameraTask::computeFocusMeasure(const Mat& img) {
    if (img.empty() || img.cols <= 0 || img.rows <= 0) {
        return 0.0;
    }

    DeviceConfig::CaptureDefaults config = getCaptureConfigSnapshot();
    int scale_factor = std::max(1, config.focusScaleFactor);
    
    int new_width = img.cols / scale_factor;
    int new_height = img.rows / scale_factor;
    if (new_width <= 0 || new_height <= 0) {
        return 0.0;
    }
    
    Mat small, gray, lap;
    cv::resize(img, small, Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);
    cvtColor(small, gray, COLOR_BGR2GRAY);
    Laplacian(gray, lap, CV_64F);
    Scalar mean_val, stddev_val;
    meanStdDev(lap, mean_val, stddev_val);
    return stddev_val.val[0] * stddev_val.val[0];
}

// -------------------- 图像级运动模糊检测 --------------------
float CameraTask::computeMotionBlurSeverity(const Mat& img) {
    if (img.empty() || img.cols < 16 || img.rows < 16) return 1.0f;

    Mat gray;
    if (img.channels() == 3) {
        cvtColor(img, gray, COLOR_BGR2GRAY);
    } else {
        gray = img;
    }

    // 缩小到合理尺寸以加快计算
    if (std::max(gray.cols, gray.rows) > 200) {
        float scale = 200.0f / std::max(gray.cols, gray.rows);
        resize(gray, gray, Size(), scale, scale, INTER_AREA);
    }

    // 1. 计算X/Y方向梯度能量
    Mat gx, gy;
    Sobel(gray, gx, CV_32F, 1, 0, 3);
    Sobel(gray, gy, CV_32F, 0, 1, 3);

    Scalar gx_mean, gx_std, gy_mean, gy_std;
    meanStdDev(gx, gx_mean, gx_std);
    meanStdDev(gy, gy_mean, gy_std);

    double energy_x = gx_std.val[0] * gx_std.val[0];
    double energy_y = gy_std.val[0] * gy_std.val[0];
    double total_energy = energy_x + energy_y;

    if (total_energy < 1.0) return 1.0f;

    // 方向不均衡度：若某一方向梯度远强于另一方向，说明运动模糊
    double dir_ratio = std::min(energy_x, energy_y) / (std::max(energy_x, energy_y) + 1e-6);
    float dir_severity = std::max(0.0f, std::min(1.0f, (float)(1.0 - dir_ratio)));

    // 2. Laplacian方差 - 综合锐度
    Mat lap;
    Laplacian(gray, lap, CV_32F);
    Scalar lap_mean_val, lap_std_val;
    meanStdDev(lap, lap_mean_val, lap_std_val);
    double lap_var = lap_std_val.val[0] * lap_std_val.val[0];
    float sharpness_severity = std::max(0.0f, std::min(1.0f, (float)((200.0 - lap_var) / 200.0)));

    // 3. 自相似移位差分 - 运动模糊拖影检测
    // 沿估计的主梯度方向移位2像素，运动模糊图像移位后差异小
    double angle = std::atan2(gy_mean.val[0], gx_mean.val[0]);
    int dx = static_cast<int>(std::round(std::cos(angle) * 2.0));
    int dy = static_cast<int>(std::round(std::sin(angle) * 2.0));
    if (dx == 0 && dy == 0) dx = 1;

    Mat shifted;
    Mat transform_mat = (Mat_<double>(2, 3) << 1, 0, dx, 0, 1, dy);
    warpAffine(gray, shifted, transform_mat, gray.size(), INTER_LINEAR, BORDER_REPLICATE);
    Mat diff;
    absdiff(gray, shifted, diff);
    double shift_diff = mean(diff).val[0];
    // 移位差分越小 = 自相关越高 = 运动模糊拖影越严重
    float trail_severity = std::max(0.0f, std::min(1.0f, (float)((16.0 - shift_diff) / 16.0)));

    // 组合
    float severity = dir_severity * 0.25f + sharpness_severity * 0.45f + trail_severity * 0.30f;
    return std::max(0.0f, std::min(1.0f, severity));
}

bool CameraTask::isFrontalFace(const std::vector<cv::Point2f>& landmarks) {
    if (landmarks.size() != 5) return false;
    DeviceConfig::CaptureDefaults config = getCaptureConfigSnapshot();
    cv::Point2f left_eye = landmarks[0];
    cv::Point2f right_eye = landmarks[1];
    cv::Point2f nose = landmarks[2];
    cv::Point2f left_mouth = landmarks[3];
    cv::Point2f right_mouth = landmarks[4];

    float dx = right_eye.x - left_eye.x;
    float dy = right_eye.y - left_eye.y;
    if (std::fabs(dx) < 1e-5f) {
        return false;
    }

    float roll = atan2(dy, dx) * 180.0 / CV_PI;

    float eye_center_x = (left_eye.x + right_eye.x) / 2.0;
    float yaw = (nose.x - eye_center_x) / dx;
    cv::Point2f mouth_center((left_mouth.x + right_mouth.x) * 0.5f,
                             (left_mouth.y + right_mouth.y) * 0.5f);
    float eye_distance = std::max(1e-5f, std::fabs(dx));
    float mouth_offset = std::fabs(mouth_center.x - eye_center_x) / eye_distance;
    float mouth_roll = std::fabs(left_mouth.y - right_mouth.y) / eye_distance;
    float nose_vertical_ratio = (nose.y - (left_eye.y + right_eye.y) * 0.5f) / eye_distance;
    float mouth_vertical_ratio = (mouth_center.y - nose.y) / eye_distance;

        return (std::fabs(roll) < config.strongFrontalMaxRoll) &&
            (std::fabs(yaw) < config.strongFrontalMaxYaw) &&
            (mouth_offset < 0.24f) &&
            (mouth_roll < 0.20f) &&
            (nose_vertical_ratio > 0.05f) &&
            (mouth_vertical_ratio > 0.08f);
}

// 新增：判断是否为侧脸
bool CameraTask::isSideFace(const std::vector<cv::Point2f>& landmarks) {
    if (landmarks.size() != 5) return false;
    cv::Point2f left_eye = landmarks[0];
    cv::Point2f right_eye = landmarks[1];
    cv::Point2f nose = landmarks[2];
    float dx = right_eye.x - left_eye.x;
    float eye_center_x = (left_eye.x + right_eye.x) / 2.0;
    float yaw = (nose.x - eye_center_x) / dx;
    // 侧脸标准
    return (fabs(yaw) >= 0.25 && fabs(yaw) < 0.6);
}

// -------------------- FPS计算 --------------------
void CameraTask::updateFPS() {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - lastFPSUpdate);
    
    if (elapsed.count() >= 1) { 
        long currentFrames = totalFrames.load();
        long framesDiff = currentFrames - framesAtLastUpdate;
        currentFPS = static_cast<double>(framesDiff) / elapsed.count();
        
        framesAtLastUpdate = currentFrames;
        lastFPSUpdate = now;
        
        auto totalElapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime);
        if (totalElapsed.count() % 10 == 0 && totalElapsed.count() > 0) {
            log_info("Algo FPS: total_processed=%ld, fps=%.2f, uptime=%lds", currentFrames, currentFPS.load(), totalElapsed.count());
        }
    }
}

bool CameraTask::enqueueCandidateEvaluation(CandidateEvalJob job) {
    DeviceConfig::CaptureDefaults config = getCaptureConfigSnapshot();
    std::lock_guard<std::mutex> lock(candidateEvalMutex);
    int pending_for_track = pendingCandidateEvalByTrack[job.trackId];
    if (pending_for_track >= config.candidatePerTrackMaxPending) {
        char detail[192];
        std::snprintf(detail, sizeof(detail), "pending=%d max=%d queue=%zu",
                      pending_for_track,
                      config.candidatePerTrackMaxPending,
                      candidateEvalQueue.size());
        logTrackReject("queue", job.trackId, "pending_limit", detail);
        return false;
    }

    auto remove_pending_for_track = [this](int track_id) {
        auto dropped_it = pendingCandidateEvalByTrack.find(track_id);
        if (dropped_it != pendingCandidateEvalByTrack.end()) {
            dropped_it->second--;
            if (dropped_it->second <= 0) {
                pendingCandidateEvalByTrack.erase(dropped_it);
            }
        }
    };

    auto drop_job_at = [&](size_t index) {
        int dropped_track_id = candidateEvalQueue[index].trackId;
        candidateEvalQueue.erase(candidateEvalQueue.begin() + index);
        remove_pending_for_track(dropped_track_id);
    };

    if (candidateEvalQueue.size() >= static_cast<size_t>(config.candidateQueueMax)) {
        size_t drop_index = candidateEvalQueue.size();

        if (pending_for_track > 0) {
            for (size_t i = 0; i < candidateEvalQueue.size(); ++i) {
                if (candidateEvalQueue[i].trackId == job.trackId) {
                    drop_index = i;
                    break;
                }
            }
        }

        if (drop_index == candidateEvalQueue.size()) {
            for (size_t i = 0; i < candidateEvalQueue.size(); ++i) {
                int queued_track_id = candidateEvalQueue[i].trackId;
                auto queued_it = pendingCandidateEvalByTrack.find(queued_track_id);
                if (queued_it != pendingCandidateEvalByTrack.end() && queued_it->second > 1) {
                    drop_index = i;
                    break;
                }
            }
        }

        if (drop_index == candidateEvalQueue.size()) {
            for (size_t i = 0; i < candidateEvalQueue.size(); ++i) {
                if (candidateEvalQueue[i].trackId != job.trackId) {
                    drop_index = i;
                    break;
                }
            }
        }

        if (drop_index == candidateEvalQueue.size()) {
            char detail[192];
            std::snprintf(detail, sizeof(detail), "queue=%zu pending=%d no_replace_slot",
                          candidateEvalQueue.size(),
                          pending_for_track);
            logTrackReject("queue", job.trackId, "queue_full", detail);
            return false;
        }

        drop_job_at(drop_index);
    }

    pendingCandidateEvalByTrack[job.trackId] = pending_for_track + 1;
    candidateEvalQueue.push_back(std::move(job));
    clearTrackReject("queue", candidateEvalQueue.back().trackId);
    candidateEvalCv.notify_one();
    return true;
}

void CameraTask::candidateEvalLoop(rknn_context faceCtx) {
    while (true) {
        CandidateEvalJob job;
        {
            std::unique_lock<std::mutex> lock(candidateEvalMutex);
            candidateEvalCv.wait(lock, [this]() {
                return !running || !candidateEvalQueue.empty();
            });

            if (!running && candidateEvalQueue.empty()) {
                break;
            }
            if (candidateEvalQueue.empty()) {
                continue;
            }

            job = std::move(candidateEvalQueue.front());
            candidateEvalQueue.pop_front();
        }

        if (!job.personRoi.empty() && job.personRoi.cols > 0 && job.personRoi.rows > 0) {
            DeviceConfig::CaptureDefaults config = getCaptureConfigSnapshot();
            double sceneBrightness = environmentBrightness.load();
            AdaptiveCaptureThresholds adaptiveThresholds =
                buildAdaptiveCaptureThresholds(config, sceneBrightness);
            Mat person_roi_resized;
            int target_width = min(config.faceInputMaxWidth, job.personRoi.cols);
            int target_height = static_cast<int>(job.personRoi.rows * target_width / (float)job.personRoi.cols);

            if (target_width > 0 && target_height > 0) {
                cv::resize(job.personRoi, person_roi_resized, Size(target_width, target_height), 0, 0, cv::INTER_LINEAR);

                std::vector<det> face_result;
                int num_faces = face_detect_run(faceCtx, person_roi_resized, face_result);
                if (num_faces <= 0 || face_result.empty()) {
                    char detail[192];
                    std::snprintf(detail, sizeof(detail), "faces=%d roi=%dx%d area=%.4f motion=%.4f",
                                  num_faces,
                                  job.personRoi.cols,
                                  job.personRoi.rows,
                                  job.areaRatio,
                                  job.motionRatio);
                    logTrackReject("eval", job.trackId, "no_face_detected", detail);
                } else {
                    int best_idx = 0;
                    for (int i = 1; i < num_faces; ++i) {
                        if (face_result[i].score > face_result[best_idx].score) {
                            best_idx = i;
                        }
                    }

                    float face_scale_x = (float)job.personRoi.cols / (float)person_roi_resized.cols;
                    float face_scale_y = (float)job.personRoi.rows / (float)person_roi_resized.rows;

                    det best_face = face_result[best_idx];
                    if (best_face.score >= config.minFaceScore && best_face.landmarks.size() >= 3) {
                        best_face.box.x *= face_scale_x;
                        best_face.box.y *= face_scale_y;
                        best_face.box.width *= face_scale_x;
                        best_face.box.height *= face_scale_y;
                        for (auto& lm : best_face.landmarks) {
                            lm.x *= face_scale_x;
                            lm.y *= face_scale_y;
                        }

                        Rect base_fbox(static_cast<int>(best_face.box.x),
                                      static_cast<int>(best_face.box.y),
                                      static_cast<int>(best_face.box.width),
                                      static_cast<int>(best_face.box.height));
                        base_fbox.x = std::max(0, std::min(base_fbox.x, job.personRoi.cols - 1));
                        base_fbox.y = std::max(0, std::min(base_fbox.y, job.personRoi.rows - 1));
                        base_fbox.width = std::min(base_fbox.width, job.personRoi.cols - base_fbox.x);
                        base_fbox.height = std::min(base_fbox.height, job.personRoi.rows - base_fbox.y);

                        if (base_fbox.width > 0 && base_fbox.height > 0) {
                            int face_short_side = std::min(base_fbox.width, base_fbox.height);
                            int face_area = base_fbox.area();
                            if (face_short_side >= config.minFaceBoxShortSide &&
                                face_area >= config.minFaceBoxArea) {
                                float person_area = static_cast<float>(job.personRoi.cols * job.personRoi.rows);
                                float face_area_ratio = face_area / std::max(1.0f, person_area);
                                float face_width_ratio = static_cast<float>(base_fbox.width) / std::max(1, job.personRoi.cols);
                                float face_center_y_ratio = static_cast<float>(base_fbox.y + base_fbox.height * 0.5f) / std::max(1, job.personRoi.rows);
                                bool face_geometry_ok =
                                    face_area_ratio >= config.faceMinAreaInPerson &&
                                    face_width_ratio >= config.faceMinWidthRatio &&
                                    face_center_y_ratio >= config.faceMinCenterYRatio &&
                                    face_center_y_ratio <= config.faceMaxCenterYRatio;

                                if (face_geometry_ok) {
                                    float margin_left = static_cast<float>(base_fbox.x);
                                    float margin_right = static_cast<float>(job.personRoi.cols - (base_fbox.x + base_fbox.width));
                                    float margin_top = static_cast<float>(base_fbox.y);
                                    float margin_bottom = static_cast<float>(job.personRoi.rows - (base_fbox.y + base_fbox.height));
                                    float margin_x_ratio = std::min(margin_left, margin_right) / std::max(1, base_fbox.width);
                                    float margin_y_ratio = std::min(margin_top, margin_bottom) / std::max(1, base_fbox.height);
                                    float min_margin_ratio = std::min(margin_x_ratio, margin_y_ratio);

                                    float face_edge_occlusion = 0.0f;
                                    if (min_margin_ratio < config.faceEdgeMinMargin) {
                                        face_edge_occlusion =
                                            (config.faceEdgeMinMargin - min_margin_ratio) / config.faceEdgeMinMargin;
                                        face_edge_occlusion = std::max(0.0f, std::min(1.0f, face_edge_occlusion));
                                    }

                                    cv::Point2f left_eye = best_face.landmarks[0];
                                    cv::Point2f right_eye = best_face.landmarks[1];
                                    float dx = right_eye.x - left_eye.x;
                                    if (std::fabs(dx) >= 1e-5f) {
                                        float dy = right_eye.y - left_eye.y;
                                        float roll = std::fabs(std::atan2(dy, dx) * 180.0f / CV_PI);
                                        float eye_center_x = (left_eye.x + right_eye.x) / 2.0f;
                                        float yaw = std::fabs((best_face.landmarks[2].x - eye_center_x) / dx);
                                        bool frontal_ok = isFrontalFace(best_face.landmarks);
                                        bool frontal_relaxed_ok = frontal_ok ||
                                            (roll < 24.0f && yaw < 0.32f && best_face.score >= 0.58f && face_edge_occlusion < 0.40f);
                                        bool weak_frontal_ok = frontal_relaxed_ok ||
                                            (roll < 32.0f && yaw < 0.48f && best_face.score >= config.minFaceScore && face_edge_occlusion < 0.72f);
                                        double reference_clarity = computeFocusMeasure(job.personRoi(base_fbox));
                                        float reference_blur_severity = computeMotionBlurSeverity(job.personRoi(base_fbox));
                                        double current_clarity = reference_clarity;
                                        float current_blur_severity = reference_blur_severity;
                                        cv::Mat fused_person_roi;
                                        const cv::Mat* capture_person_roi = &job.personRoi;
                                        bool multi_frame_fused = false;
                                        int fusion_frame_count = 0;
                                        float fusion_mean_similarity = 0.0f;
                                        bool fusion_enabled =
                                            adaptiveThresholds.lowLightStrength >= kMultiFrameFusionLowLightMinStrength &&
                                            !job.fusionHistory.empty() &&
                                            job.motionRatio <= adaptiveThresholds.maxMotionRejectRatio;
                                        if (fusion_enabled) {
                                            MultiFrameFusionResult fusion_result = fuseTrackHistoryPersonRoi(job.personRoi,
                                                                                                             job.fusionHistory,
                                                                                                             base_fbox,
                                                                                                             adaptiveThresholds.lowLightStrength,
                                                                                                             job.motionRatio);
                                            if (!fusion_result.fused.empty() && fusion_result.acceptedFrames >= 2) {
                                                double fused_clarity = computeFocusMeasure(fusion_result.fused(base_fbox));
                                                float fused_blur_severity = computeMotionBlurSeverity(fusion_result.fused(base_fbox));
                                                bool fusion_better =
                                                    fused_clarity > reference_clarity * 1.06 + 3.0 ||
                                                    (fused_clarity >= reference_clarity * 0.99 &&
                                                     fused_blur_severity + 0.06f < reference_blur_severity) ||
                                                    (fused_blur_severity + 0.10f < reference_blur_severity &&
                                                     fused_clarity >= reference_clarity * 0.95);
                                                if (fusion_better) {
                                                    fused_person_roi = std::move(fusion_result.fused);
                                                    capture_person_roi = &fused_person_roi;
                                                    current_clarity = fused_clarity;
                                                    current_blur_severity = fused_blur_severity;
                                                    multi_frame_fused = true;
                                                    fusion_frame_count = fusion_result.acceptedFrames;
                                                    fusion_mean_similarity = fusion_result.meanSimilarity;
                                                    log_debug("Track %d multi-frame fusion used: frames=%d sim=%.2f clarity=%.1f->%.1f blur=%.2f->%.2f",
                                                              job.trackId,
                                                              fusion_result.acceptedFrames,
                                                              fusion_result.meanSimilarity,
                                                              reference_clarity,
                                                              fused_clarity,
                                                              reference_blur_severity,
                                                              fused_blur_severity);
                                                }
                                            }
                                        }
                                        bool motion_gate_ok = job.motionRatio < adaptiveThresholds.maxMotionRatio;
                                        bool strong_candidate_ok =
                                            (!config.requireFrontalFace || frontal_relaxed_ok) &&
                                            current_clarity > adaptiveThresholds.minClarity &&
                                            current_blur_severity < adaptiveThresholds.maxBlurSeverity &&
                                            yaw < config.maxYaw &&
                                            motion_gate_ok;
                                        bool fallback_candidate_ok =
                                            (!config.requireFrontalFace || weak_frontal_ok) &&
                                            current_clarity > adaptiveThresholds.fallbackMinClarity &&
                                            current_blur_severity < adaptiveThresholds.fallbackMaxBlurSeverity &&
                                            yaw < config.fallbackMaxYaw &&
                                            face_edge_occlusion < config.fallbackMaxFaceEdgeOcclusion &&
                                            motion_gate_ok;

                                        if (strong_candidate_ok || fallback_candidate_ok) {
                                            int crop_w = std::max(1, static_cast<int>(base_fbox.width * config.headshotExpandRatio));
                                            int crop_h = std::max(1, static_cast<int>(base_fbox.height * config.headshotExpandRatio));
                                            int crop_cx = base_fbox.x + base_fbox.width / 2;
                                            int crop_cy = base_fbox.y + base_fbox.height / 2 + static_cast<int>(base_fbox.height * config.headshotDownShift);
                                            int crop_x = crop_cx - crop_w / 2;
                                            int crop_y = crop_cy - crop_h / 2;

                                            crop_x = std::max(0, std::min(job.personRoi.cols - crop_w, crop_x));
                                            crop_y = std::max(0, std::min(job.personRoi.rows - crop_h, crop_y));
                                            Rect fbox(crop_x, crop_y,
                                                      std::min(crop_w, job.personRoi.cols - crop_x),
                                                      std::min(crop_h, job.personRoi.rows - crop_y));

                                            if (fbox.width > 0 && fbox.height > 0) {
                                                float crop_margin_left = static_cast<float>(base_fbox.x - fbox.x) / std::max(1, base_fbox.width);
                                                float crop_margin_right = static_cast<float>((fbox.x + fbox.width) - (base_fbox.x + base_fbox.width)) / std::max(1, base_fbox.width);
                                                float crop_margin_top = static_cast<float>(base_fbox.y - fbox.y) / std::max(1, base_fbox.height);
                                                float crop_margin_bottom = static_cast<float>((fbox.y + fbox.height) - (base_fbox.y + base_fbox.height)) / std::max(1, base_fbox.height);
                                                float crop_min_margin = std::min(std::min(crop_margin_left, crop_margin_right),
                                                                                 std::min(crop_margin_top, crop_margin_bottom));
                                                float required_crop_margin = strong_candidate_ok ?
                                                    config.headshotMinFaceMargin :
                                                    config.fallbackHeadshotMinFaceMargin;
                                                if (crop_min_margin >= required_crop_margin) {
                                                    Mat face_aligned = (*capture_person_roi)(fbox).clone();
                                                    int upper_body_w = std::min(job.personRoi.cols,
                                                        std::max(static_cast<int>(base_fbox.width * config.upperBodyWidthFaceRatio),
                                                                 static_cast<int>(job.personRoi.cols * config.upperBodyMinWidthRatio)));
                                                    int upper_body_h = std::min(job.personRoi.rows,
                                                        std::max(static_cast<int>(base_fbox.height * config.upperBodyHeightFaceRatio),
                                                                 static_cast<int>(job.personRoi.rows * config.upperBodyMinHeightRatio)));
                                                    int upper_body_cx = base_fbox.x + base_fbox.width / 2;
                                                    int upper_body_cy = base_fbox.y + static_cast<int>(base_fbox.height * config.upperBodyCenterYRatio);
                                                    int upper_body_x = std::max(0, std::min(job.personRoi.cols - upper_body_w, upper_body_cx - upper_body_w / 2));
                                                    int upper_body_y = std::max(0, std::min(job.personRoi.rows - upper_body_h,
                                                        upper_body_cy - static_cast<int>(upper_body_h / config.upperBodyTopDivisor)));
                                                    cv::Rect upper_body_box(
                                                        upper_body_x,
                                                        upper_body_y,
                                                        std::min(upper_body_w, job.personRoi.cols - upper_body_x),
                                                        std::min(upper_body_h, job.personRoi.rows - upper_body_y));
                                                    cv::Mat person_aligned = (*capture_person_roi)(upper_body_box).clone();
                                                    float quality_weight, area_weight;
                                                    if (yaw < 0.15f) {
                                                        quality_weight = 0.8f;
                                                        area_weight = 0.35f;
                                                    } else if (yaw < 0.30f) {
                                                        float ratio = (yaw - 0.15f) / 0.15f;
                                                        quality_weight = 0.8f - ratio * 0.3f;
                                                        area_weight = 0.35f - ratio * 0.15f;
                                                    } else if (yaw < 0.50f) {
                                                        float ratio = (yaw - 0.30f) / 0.20f;
                                                        quality_weight = 0.5f - ratio * 0.25f;
                                                        area_weight = 0.2f - ratio * 0.12f;
                                                    } else {
                                                        float ratio = (yaw - 0.50f) / 0.20f;
                                                        quality_weight = 0.25f - ratio * 0.15f;
                                                        area_weight = 0.08f - ratio * 0.05f;
                                                    }

                                                    float area_score = 1.0f / (1.0f + std::fabs(job.areaRatio - config.areaScoreTargetRatio) /
                                                                                        std::max(1e-6f, config.areaScoreTargetRatio));
                                                    float clarity_norm = static_cast<float>(std::min(1.8, current_clarity / std::max(1.0, adaptiveThresholds.minClarity)));
                                                    float person_occ_norm = job.personOcclusion / std::max(1e-6f, config.maxPersonOcclusion);
                                                    person_occ_norm = std::max(0.0f, std::min(1.5f, person_occ_norm));
                                                    float face_edge_occ_norm = face_edge_occlusion / std::max(1e-6f, config.maxFaceEdgeOcclusion);
                                                    face_edge_occ_norm = std::max(0.0f, std::min(1.5f, face_edge_occ_norm));
                                                    float occlusion_penalty = person_occ_norm * 0.7f + face_edge_occ_norm * 0.3f;
                                                    float motion_penalty = std::min(1.25f, job.motionRatio / std::max(1e-6f, adaptiveThresholds.maxMotionRatio));
                                                    float blur_severity_penalty = std::min(1.25f, current_blur_severity / std::max(1e-6f, adaptiveThresholds.maxBlurSeverity));
                                                    float candidate_penalty = strong_candidate_ok ? 0.0f : config.fallbackScorePenalty;
                                                    float clarity_gain = std::max(0.0f, clarity_norm - 1.0f);
                                                    float face_confidence_bonus = std::min(1.0f, std::max(0.0f, best_face.score)) * 90.0f;
                                                    float frontal_bonus = frontal_ok ? 135.0f : (frontal_relaxed_ok ? 75.0f : 20.0f);
                                                    float crop_margin_bonus =
                                                        std::min(1.6f, crop_min_margin / std::max(1e-6f, required_crop_margin)) * 75.0f;
                                                    float face_size_bonus =
                                                        std::min(1.8f, face_area_ratio / std::max(1e-6f, config.faceMinAreaInPerson)) * 45.0f;
                                                    float yaw_penalty =
                                                        std::min(1.25f, yaw / std::max(0.10f, strong_candidate_ok ? config.maxYaw : config.fallbackMaxYaw)) * 60.0f;

                                                    Track::FrameData frame_data;
                                                    frame_data.score = current_clarity * (0.55f + quality_weight * 0.20f) +
                                                                       clarity_norm * 220.0f +
                                                                       clarity_gain * 160.0f +
                                                                       frontal_bonus +
                                                                       face_confidence_bonus +
                                                                       crop_margin_bonus +
                                                                       face_size_bonus +
                                                                       area_score * 1000 * area_weight * 0.22f -
                                                                       occlusion_penalty * config.occlusionScorePenalty * 1.15f -
                                                                       motion_penalty * config.motionScorePenalty * 1.30f -
                                                                       blur_severity_penalty * config.blurSeverityScorePenalty * 1.45f -
                                                                       yaw_penalty -
                                                                       candidate_penalty;
                                                    frame_data.person_roi = person_aligned;
                                                    frame_data.face_roi = face_aligned;
                                                    frame_data.has_face = true;
                                                    frame_data.is_frontal = frontal_ok;
                                                    frame_data.face_pose_level = frontal_ok ? 2 : (frontal_relaxed_ok ? 1 : 0);
                                                    frame_data.strong_candidate = strong_candidate_ok;
                                                    frame_data.yaw_abs = yaw;
                                                    frame_data.clarity = current_clarity;
                                                    frame_data.area_ratio = job.areaRatio;
                                                    frame_data.person_occlusion = job.personOcclusion;
                                                    frame_data.face_edge_occlusion = face_edge_occlusion;
                                                    frame_data.motion_ratio = job.motionRatio;
                                                    frame_data.blur_severity = current_blur_severity;
                                                    add_frame_candidate(job.trackId, frame_data);
                                                    clearTrackReject("eval", job.trackId);

                                                    if (multi_frame_fused) {
                                                        log_debug("Track %d fused candidate accepted: frames=%d sim=%.2f clarity=%.1f blur=%.2f yaw=%.2f edge=%.2f",
                                                                  job.trackId,
                                                                  fusion_frame_count,
                                                                  fusion_mean_similarity,
                                                                  current_clarity,
                                                                  current_blur_severity,
                                                                  yaw,
                                                                  face_edge_occlusion);
                                                    } else if (!strong_candidate_ok) {
                                                        log_debug("Track %d fallback candidate accepted: clarity=%.1f blur=%.2f yaw=%.2f edge=%.2f",
                                                                  job.trackId,
                                                                  current_clarity,
                                                                  current_blur_severity,
                                                                  yaw,
                                                                  face_edge_occlusion);
#if 0
                                                    if (!strong_candidate_ok) {
                                                        log_debug("Track %d 添加兜底候选帧: clarity=%.1f yaw=%.2f occ=%.2f",
                                                                  job.trackId,
                                                                  current_clarity,
                                                                  yaw,
                                                                  face_edge_occlusion);
                                                    }
#endif
                                                    }
                                                } else {
                                                    char detail[224];
                                                    std::snprintf(detail, sizeof(detail),
                                                                  "crop_margin=%.2f need=%.2f yaw=%.2f clarity=%.1f",
                                                                  crop_min_margin,
                                                                  required_crop_margin,
                                                                  yaw,
                                                                  current_clarity);
                                                    logTrackReject("eval", job.trackId, "headshot_margin", detail);
                                                }
                                            } else {
                                                char detail[160];
                                                std::snprintf(detail, sizeof(detail), "invalid_headshot_box=%dx%d", fbox.width, fbox.height);
                                                logTrackReject("eval", job.trackId, "headshot_box_invalid", detail);
                                            }
                                        } else {
                                            char detail[320];
                                            std::snprintf(detail, sizeof(detail),
                                                          "frontal=%d relaxed=%d weak=%d yaw=%.2f clarity=%.1f edge=%.2f blur=%.2f motion=%.4f bright=%.1f ll=%.2f",
                                                          frontal_ok ? 1 : 0,
                                                          frontal_relaxed_ok ? 1 : 0,
                                                          weak_frontal_ok ? 1 : 0,
                                                          yaw,
                                                          current_clarity,
                                                          face_edge_occlusion,
                                                          current_blur_severity,
                                                          job.motionRatio,
                                                          sceneBrightness,
                                                          adaptiveThresholds.lowLightStrength);
                                            const char* reason = "quality_gate";
                                            if (config.requireFrontalFace && !weak_frontal_ok) {
                                                reason = "non_frontal";
                                            } else if (!motion_gate_ok) {
                                                reason = "motion_soft_large";
                                            } else if (current_clarity <= adaptiveThresholds.fallbackMinClarity) {
                                                reason = "clarity_low";
                                            } else if (current_blur_severity >= adaptiveThresholds.fallbackMaxBlurSeverity) {
                                                reason = "motion_blur";
                                            } else if (yaw >= config.fallbackMaxYaw) {
                                                reason = "yaw_large";
                                            } else if (face_edge_occlusion >= config.fallbackMaxFaceEdgeOcclusion) {
                                                reason = "edge_occlusion";
                                            }
                                            logTrackReject("eval", job.trackId, reason, detail);
                                        }
                                    } else {
                                        char detail[160];
                                        std::snprintf(detail, sizeof(detail), "eye_dx=%.5f", dx);
                                        logTrackReject("eval", job.trackId, "eye_distance_small", detail);
                                    }
                                } else {
                                    char detail[224];
                                    std::snprintf(detail, sizeof(detail),
                                                  "area_ratio=%.3f width_ratio=%.3f center_y=%.3f",
                                                  face_area_ratio,
                                                  face_width_ratio,
                                                  face_center_y_ratio);
                                    logTrackReject("eval", job.trackId, "face_geometry", detail);
                                }
                            } else {
                                char detail[224];
                                std::snprintf(detail, sizeof(detail), "short=%d min_short=%d area=%d min_area=%d",
                                              face_short_side,
                                              config.minFaceBoxShortSide,
                                              face_area,
                                              config.minFaceBoxArea);
                                logTrackReject("eval", job.trackId, "face_box_small", detail);
                            }
                        } else {
                            char detail[160];
                            std::snprintf(detail, sizeof(detail), "base_fbox=%dx%d", base_fbox.width, base_fbox.height);
                            logTrackReject("eval", job.trackId, "face_box_invalid", detail);
                        }
                    } else {
                        char detail[192];
                        std::snprintf(detail, sizeof(detail), "score=%.2f min=%.2f landmarks=%zu",
                                      best_face.score,
                                      config.minFaceScore,
                                      best_face.landmarks.size());
                        logTrackReject("eval", job.trackId,
                                       best_face.landmarks.size() < 3 ? "landmarks_missing" : "face_score_low",
                                       detail);
                    }
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock(candidateEvalMutex);
            auto it = pendingCandidateEvalByTrack.find(job.trackId);
            if (it != pendingCandidateEvalByTrack.end()) {
                it->second--;
                if (it->second <= 0) {
                    pendingCandidateEvalByTrack.erase(it);
                }
            }
        }
    }
}

void CameraTask::run() {
    log_info("CameraTask: starting camera task...");
    log_info("CameraTask: working directory: %s", getcwd(NULL, 0));
    log_info("CameraTask: person model path: %s", personModelPath.c_str());
    log_info("CameraTask: face model path: %s", faceModelPath.c_str());
    
    // 检查模型文件是否存在
    FILE* fp = fopen(personModelPath.c_str(), "r");
    if (fp) {
        fclose(fp);
        log_info("CameraTask: person model file exists");
    } else {
        log_error("CameraTask: person model file NOT found: %s", personModelPath.c_str());
        return;
    }
    
    fp = fopen(faceModelPath.c_str(), "r");
    if (fp) {
        fclose(fp);
        log_info("CameraTask: face model file exists");
    } else {
        log_error("CameraTask: face model file NOT found: %s", faceModelPath.c_str());
        return;
    }
    
    log_info("CameraTask: loading person model...");
    fflush(stdout);  // 强制刷新输出
    fflush(stderr);
    
    auto start_time = std::chrono::steady_clock::now();
    rknn_context personCtx, faceCtx;
    
    int ret = person_detect_init(&personCtx, personModelPath.c_str());
    
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start_time);
    
    if (ret != 0) {
        log_error("CameraTask: person_detect_init failed with code %d (model: %s, time: %lds)", 
                  ret, personModelPath.c_str(), elapsed.count());
        return;
    }
    log_info("CameraTask: person model loaded successfully in %ld seconds", elapsed.count());
    
    log_info("CameraTask: loading face model...");
    fflush(stdout);
    fflush(stderr);
    
    start_time = std::chrono::steady_clock::now();
    ret = face_detect_init(&faceCtx, faceModelPath.c_str());
    elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start_time);
    
    if (ret != 0) {
        log_error("CameraTask: face_detect_init failed with code %d (model: %s, time: %lds)", 
                  ret, faceModelPath.c_str(), elapsed.count());
        person_detect_release(personCtx);
        return;
    }
    log_info("CameraTask: face model loaded successfully in %ld seconds", elapsed.count());
    sort_init();
    lastTrackCenters.clear();
    trackPersonRoiHistory.clear();
    trackApproachStates.clear();
    reportedPersonIds.clear();
    candidateRoundRobinOffset = 0;
    hadPersonsInScene = false;

    set_upload_callback([this](const cv::Mat& img, int id, const std::string& type) {
        if (uploadCallback) {
            uploadCallback(img, id, type);
        }
    }, &capturedPersonIds, &capturedFaceIds);
    
    log_info("CameraTask: initializing camera (index=%d, resolution=%dx%d)...", 
             cameraIndex, CAMERA_WIDTH, CAMERA_HEIGHT);
    
    if (mipicamera_init(cameraIndex, CAMERA_WIDTH, CAMERA_HEIGHT, 0) != 0) {
        log_error("CameraTask: camera init failed (index=%d)", cameraIndex);
        person_detect_release(personCtx);
        face_detect_release(faceCtx);
        return;
    }
    cameraOpened = true;
    mipicamera_set_format(cameraIndex, CAMERA_FORMAT);
    log_info("CameraTask: camera initialized successfully (format=%s)",
             CAMERA_FORMAT == RK_FORMAT_YCbCr_420_SP ? "NV12" :
             CAMERA_FORMAT == RK_FORMAT_BGR_888 ? "BGR888" :
             CAMERA_FORMAT == RK_FORMAT_RGB_888 ? "RGB888" : "UNKNOWN");
    log_info("CameraTask: camera initialized successfully (format=%s)",
             CAMERA_FORMAT == RK_FORMAT_YCbCr_420_SP ? "NV12" :
             CAMERA_FORMAT == RK_FORMAT_BGR_888 ? "BGR888" :
             CAMERA_FORMAT == RK_FORMAT_RGB_888 ? "RGB888" : "UNKNOWN");
   /*
   if (usbcamera_init(cameraIndex, CAMERA_WIDTH, CAMERA_HEIGHT, 0) != 0) {
        log_error("CameraTask: USB camera init failed");
        person_detect_release(personCtx);
        face_detect_release(faceCtx);
        return;
    }
   */

    log_info("CameraTask: starting capture/inference loops...");
    reportedPersonIds.clear();
    {
        std::lock_guard<std::mutex> lock(frameMutex);
        latestFrame.release();
        latestFrameSeq = 0;
        consumedFrameSeq = 0;
    }
    {
        std::lock_guard<std::mutex> lock(candidateEvalMutex);
        candidateEvalQueue.clear();
        pendingCandidateEvalByTrack.clear();
    }
    candidateWorker = std::thread(&CameraTask::candidateEvalLoop, this, faceCtx);
    captureWorker = std::thread(&CameraTask::captureLoop, this);

    while (running) {
        Mat frame;
        {
            std::unique_lock<std::mutex> lock(frameMutex);
            frameCv.wait_for(lock, std::chrono::milliseconds(100), [this]() {
                return !running || latestFrameSeq != consumedFrameSeq;
            });

            if (!running && latestFrameSeq == consumedFrameSeq) {
                break;
            }
            if (latestFrameSeq == consumedFrameSeq) {
                continue;
            }

            frame = latestFrame;
            consumedFrameSeq = latestFrameSeq;
        }

        if (frame.empty() || frame.cols <= 0 || frame.rows <= 0) {
            log_error("CameraTask: invalid frame dimensions (width=%d, height=%d)", frame.cols, frame.rows);
            continue;
        }
        
        // 帧数统计
        totalFrames++;
        updateFPS();
        
        processFrame(frame, personCtx);
    }

    if (captureWorker.joinable()) {
        captureWorker.join();
    }
    candidateEvalCv.notify_all();
    if (candidateWorker.joinable()) {
        candidateWorker.join();
    }

    log_info("CameraTask: inference loop exited, cleaning up...");
    if (cameraOpened.exchange(false)) {
        mipicamera_exit(cameraIndex);
    }
    person_detect_release(personCtx);
    face_detect_release(faceCtx);
    log_info("CameraTask: cleanup completed");
}

void CameraTask::captureLoop() {
    std::vector<unsigned char> buffer(IMAGE_SIZE);
    auto captureFpsWindowStart = std::chrono::steady_clock::now();
    long captureFramesInWindow = 0;
    int brightnessSampleCounter = 0;
    int whiteCandidateHits = 0;
    int blackCandidateHits = 0;
    double lastBrightnessRaw = 0.0;
    double filteredBrightness = -1.0;

    // 默认先使用白片，后续再按亮度阈值自动切换。
    g_lastIrCutSwitchTime = std::chrono::steady_clock::now() - std::chrono::seconds(3600);
    switchIrCutWhite();

    while (running) {
        if (mipicamera_getframe(cameraIndex, reinterpret_cast<char*>(buffer.data())) != 0) {
            if (!running) {
                break;
            }
            continue;
        }

        Mat frame(CAMERA_HEIGHT, CAMERA_WIDTH, CV_8UC3, buffer.data());
        if (frame.empty() || frame.cols <= 0 || frame.rows <= 0) {
            continue;
        }

        DeviceConfig::CaptureDefaults capture = getCaptureConfigSnapshot();
        DeviceConfig::BrightnessBoostConfig boostConfig = getBrightnessBoostConfigSnapshot();

        brightnessSampleCounter++;
        int brightnessSampleInterval = std::max(1, capture.brightnessSampleInterval);
        if (brightnessSampleCounter % brightnessSampleInterval == 0) {
            double brightnessRaw = computeSceneBrightnessFast(frame);
            lastBrightnessRaw = brightnessRaw;
            filteredBrightness = updateFilteredBrightness(filteredBrightness, brightnessRaw);
            environmentBrightness = filteredBrightness;

            auto now = std::chrono::steady_clock::now();
            auto sinceLastSwitch = std::chrono::duration_cast<std::chrono::seconds>(now - g_lastIrCutSwitchTime).count();
            const double blackThreshold = brightnessBlackThreshold.load();
            if (sinceLastSwitch < kIrCutSettleAfterSwitchSec) {
                whiteCandidateHits = 0;
                blackCandidateHits = 0;
            } else if (filteredBrightness >= capture.brightnessWhiteThreshold) {
                whiteCandidateHits++;
                blackCandidateHits = 0;
                if (whiteCandidateHits >= kIrCutConsecutiveHits) {
                    switchIrCutWhite();
                    whiteCandidateHits = 0;
                    blackCandidateHits = 0;
                }
            } else if (filteredBrightness <= blackThreshold) {
                blackCandidateHits++;
                whiteCandidateHits = 0;
                if (blackCandidateHits >= kIrCutConsecutiveHits) {
                    switchIrCutBlack();
                    whiteCandidateHits = 0;
                    blackCandidateHits = 0;
                }
            } else {
                whiteCandidateHits = 0;
                blackCandidateHits = 0;
            }
        }

        // ─── 仅对实际推送给推理线程的帧做亮度补偿，丢弃帧跳过 ───
        bool published = false;
        bool boosted = false;
        {
            std::lock_guard<std::mutex> lock(frameMutex);
            if (latestFrameSeq == consumedFrameSeq) {
                latestFrame = frame.clone();
                if (filteredBrightness > boostConfig.boostMinFloor
                    && filteredBrightness < boostConfig.boostThreshold) {
                    boosted = applyBrightnessBoost(latestFrame, filteredBrightness, boostConfig);
                }
                latestFrameSeq++;
                published = true;
            }
        }
        if (published) {
            frameCv.notify_one();
        }

        captureFramesInWindow++;
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - captureFpsWindowStart).count();
        if (elapsed >= 5) {
            double captureFps = static_cast<double>(captureFramesInWindow) / static_cast<double>(elapsed);
            log_info("Capture FPS: fps=%.2f, brightness=%.1f(raw=%.1f), ircut=%s, black_th=%.1f, boost=%s",
                     captureFps,
                     environmentBrightness.load(),
                     lastBrightnessRaw,
                     irCutModeToString(g_irCutMode),
                     brightnessBlackThreshold.load(),
                     boosted ? "ON" : "off");
            captureFpsWindowStart = now;
            captureFramesInWindow = 0;
        }
    }
}

void CameraTask::captureSnapshot() {
    if (!cameraOpened) {
        log_warn("CameraTask: snapshot ignored, camera is not opened");
        return;
    }

    std::vector<unsigned char> buffer(IMAGE_SIZE);
    if (mipicamera_getframe(cameraIndex, reinterpret_cast<char*>(buffer.data())) == 0) {
        cv::Mat frame(CAMERA_HEIGHT, CAMERA_WIDTH, CV_8UC3, buffer.data());
        if (!frame.empty()) {
            if (uploadCallback) {
                uploadCallback(frame.clone(), 0, "manual");
                log_info("CameraTask: snapshot uploaded");
            }
        } else {
            log_error("CameraTask: snapshot empty frame");
        }
    } else {
        log_error("CameraTask: snapshot failed to get frame");
    }
}


void CameraTask::processFrame(const Mat& frame, rknn_context personCtx) {
    static int personDetectCounter = 0;
    static std::vector<Track> cachedTracks;
    DeviceConfig::CaptureDefaults config = getCaptureConfigSnapshot();
    double sceneBrightness = environmentBrightness.load();
    AdaptiveCaptureThresholds adaptiveThresholds =
        buildAdaptiveCaptureThresholds(config, sceneBrightness);

    const float inv_diag_720p = 1.0f / std::sqrt((float)IMAGE_WIDTH * IMAGE_WIDTH + (float)IMAGE_HEIGHT * IMAGE_HEIGHT);

    float scale_x = (float)IMAGE_WIDTH / (float)CAMERA_WIDTH;   // 1280/3840 = 0.333
    float scale_y = (float)IMAGE_HEIGHT / (float)CAMERA_HEIGHT; // 720/2160 = 0.333
    
    /* //经测试硬件RGA加速反而更慢（加锁开销大），继续OpenCV软件缩放
    // 使用RGA硬件加速进行缩放 (4K -> 720p)
    Image src_img, dst_img;
    
    // 源图像设置 (4K BGR)
    src_img.width = CAMERA_WIDTH;
    src_img.height = CAMERA_HEIGHT;
    src_img.hor_stride = CAMERA_WIDTH;
    src_img.ver_stride = CAMERA_HEIGHT;
    src_img.fmt = RK_FORMAT_BGR_888;
    src_img.rotation = HAL_TRANSFORM_ROT_0;
    src_img.pBuf = frame.data;
    
    // 目标图像设置 (720p BGR)
    dst_img.width = IMAGE_WIDTH;
    dst_img.height = IMAGE_HEIGHT;
    dst_img.hor_stride = IMAGE_WIDTH;
    dst_img.ver_stride = IMAGE_HEIGHT;
    dst_img.fmt = RK_FORMAT_BGR_888;
    dst_img.rotation = HAL_TRANSFORM_ROT_0;
    dst_img.pBuf = resized_buffer_720p;
    
    if (srcImg_ConvertTo_dstImg(&dst_img, &src_img) != 0) {
        log_error("RGA resize failed, fallback to OpenCV");
        Mat resized_frame;
        cv::resize(frame, resized_frame, Size(IMAGE_WIDTH, IMAGE_HEIGHT), 0, 0, cv::INTER_LINEAR);
        memcpy(resized_buffer_720p, resized_frame.data, IMAGE_WIDTH * IMAGE_HEIGHT * 3);
    }
    Mat resized_frame(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, resized_buffer_720p);
    */
    
    Mat resized_frame;
    cv::resize(frame, resized_frame, Size(IMAGE_WIDTH, IMAGE_HEIGHT), 0, 0, cv::INTER_LINEAR);

    int personDetectInterval = std::max(1, config.personDetectInterval);
    bool runPersonDetect = (personDetectCounter % personDetectInterval == 0) || cachedTracks.empty();
    personDetectCounter++;

    if (runPersonDetect) {
        detect_result_group_t detect_result_group;
        person_detect_run(personCtx, resized_frame, &detect_result_group);

        vector<Detection> dets;
        dets.reserve(detect_result_group.count);
        for (int i = 0; i < detect_result_group.count; i++) {
            detect_result_t& d = detect_result_group.results[i];
            if (d.prop < 0.7f) continue;

            Rect roi_720p(max(0, d.box.left), max(0, d.box.top),
                          min(IMAGE_WIDTH - 1, d.box.right) - max(0, d.box.left),
                          min(IMAGE_HEIGHT - 1, d.box.bottom) - max(0, d.box.top));
            if (roi_720p.width <= 0 || roi_720p.height <= 0) continue;

            Detection det;
            det.roi = resized_frame(roi_720p);
            det.x1 = roi_720p.x;
            det.y1 = roi_720p.y;
            det.x2 = roi_720p.x + roi_720p.width;
            det.y2 = roi_720p.y + roi_720p.height;
            det.prop = d.prop;
            dets.push_back(det);
        }

        nmsDetections(dets, 0.55f);
        cachedTracks = sort_update(dets);
    }

    vector<Track> tracks = cachedTracks;
    std::unordered_set<int> activeTrackIds;

    std::unordered_map<int, cv::Rect> trackBoxes720p;
    trackBoxes720p.reserve(tracks.size());
    for (const auto& tr : tracks) {
        cv::Rect2f stable_box = selectTrackRect720p(tr);
        cv::Rect box((int)stable_box.x, (int)stable_box.y, (int)stable_box.width, (int)stable_box.height);
        if (box.width > 0 && box.height > 0) {
            trackBoxes720p.emplace(tr.id, box);
        }
    }

    std::unordered_map<int, float> trackOcclusionRatio;
    trackOcclusionRatio.reserve(trackBoxes720p.size());
    for (const auto& kv_a : trackBoxes720p) {
        float max_overlap = 0.0f;
        for (const auto& kv_b : trackBoxes720p) {
            if (kv_a.first == kv_b.first) {
                continue;
            }
            float overlap = rect_overlap_ratio_on_a(kv_a.second, kv_b.second);
            if (overlap > max_overlap) {
                max_overlap = overlap;
            }
        }
        trackOcclusionRatio[kv_a.first] = max_overlap;
    }

    if (!tracks.empty()) {
        candidateRoundRobinOffset %= tracks.size();
    } else {
        candidateRoundRobinOffset = 0;
    }

    size_t track_count = tracks.size();
    for (size_t index = 0; index < track_count; ++index) {
        auto& t = tracks[(candidateRoundRobinOffset + index) % track_count];
        activeTrackIds.insert(t.id);

        cv::Rect2f stable_bbox_720p = selectTrackRect720p(t);
        Rect bbox_720p((int)stable_bbox_720p.x, (int)stable_bbox_720p.y,
                       (int)stable_bbox_720p.width, (int)stable_bbox_720p.height);
        if (bbox_720p.width <=0 || bbox_720p.height <=0) continue;

        cv::Point2f curr_center_720p(bbox_720p.x + bbox_720p.width * 0.5f,
                                     bbox_720p.y + bbox_720p.height * 0.5f);
        float motion_ratio = 0.0f;
        auto prev_it = lastTrackCenters.find(t.id);
        if (prev_it != lastTrackCenters.end()) {
            float pixel_motion = cv::norm(curr_center_720p - prev_it->second);
            motion_ratio = pixel_motion * inv_diag_720p;
        }
        lastTrackCenters[t.id] = curr_center_720p;

        if (t.hits < config.minTrackHits) {
            char detail[160];
            std::snprintf(detail, sizeof(detail), "hits=%d min_hits=%d", t.hits, config.minTrackHits);
            logTrackReject("gate", t.id, "track_unstable", detail);
            continue;
        }

        if (reportedPersonIds.find(t.id) == reportedPersonIds.end()) {
            reportedPersonIds.insert(t.id);
            if (personEventCallback) {
                personEventCallback(t.id, "person_appeared");
            }
        }
        
        // 将720p的bbox映射回4K坐标系，用于从原图截取高质量ROI
        int orig_x = static_cast<int>(bbox_720p.x / scale_x);
        int orig_y = static_cast<int>(bbox_720p.y / scale_y);
        int orig_width = static_cast<int>(bbox_720p.width / scale_x);
        int orig_height = static_cast<int>(bbox_720p.height / scale_y);
        
        orig_x = max(0, min(CAMERA_WIDTH - orig_width, orig_x));
        orig_y = max(0, min(CAMERA_HEIGHT - orig_height, orig_y));
        orig_width = min(CAMERA_WIDTH - orig_x, orig_width);
        orig_height = min(CAMERA_HEIGHT - orig_y, orig_height);
        
        if (orig_width <= 0 || orig_height <= 0) continue;
        
        int expand_x = static_cast<int>(orig_width * config.personContextExpandX);
        int expand_top = static_cast<int>(orig_height * config.personContextExpandTop);
        int expand_bottom = static_cast<int>(orig_height * config.personContextExpandBottom);
        int expanded_x = std::max(0, orig_x - expand_x);
        int expanded_y = std::max(0, orig_y - expand_top);
        int expanded_right = std::min(CAMERA_WIDTH, orig_x + orig_width + expand_x);
        int expanded_bottom = std::min(CAMERA_HEIGHT, orig_y + orig_height + expand_bottom);
        Rect bbox_4k(expanded_x,
                 expanded_y,
                 std::max(1, expanded_right - expanded_x),
                 std::max(1, expanded_bottom - expanded_y));

        Mat person_roi = frame(bbox_4k);
        if (person_roi.empty() || person_roi.cols <= 0 || person_roi.rows <= 0) {
            char detail[192];
            std::snprintf(detail, sizeof(detail), "roi=%dx%d bbox4k=%dx%d",
                          person_roi.cols,
                          person_roi.rows,
                          bbox_4k.width,
                          bbox_4k.height);
            logTrackReject("gate", t.id, "person_roi_invalid", detail);
            continue;
        }

        std::vector<cv::Mat> fusion_history;
        auto history_it = trackPersonRoiHistory.find(t.id);
        if (history_it != trackPersonRoiHistory.end()) {
            fusion_history.reserve(history_it->second.size());
            for (const auto& hist_roi : history_it->second) {
                if (!hist_roi.empty()) {
                    fusion_history.push_back(hist_roi.clone());
                }
            }
        }
        auto& roi_history = trackPersonRoiHistory[t.id];
        roi_history.push_back(person_roi.clone());
        while (roi_history.size() > kMultiFrameFusionHistorySize) {
            roi_history.pop_front();
        }

        float current_area_4k = bbox_4k.width * bbox_4k.height;
        float area_ratio = current_area_4k / (CAMERA_WIDTH * CAMERA_HEIGHT);

        float area_trend_ratio = 0.0f;
        if (t.bbox_history.size() >= 5) {
            float area_now = t.bbox_history.back();
            float area_prev = t.bbox_history[t.bbox_history.size() - 5];
            area_trend_ratio = (area_now - area_prev) / (area_prev + 1e-6f);

            // 先更新接近状态，再用于后续抓拍门控
            if (area_trend_ratio > config.approachRatioPos) {
                t.is_approaching = true;
            } else if (area_trend_ratio < config.approachRatioNeg) {
                t.is_approaching = false;
            }
        }

        bool near_ok = area_ratio >= config.nearAreaRatio;
        bool approach_ok = t.is_approaching || near_ok || !config.requireApproach;
        auto& approachState = trackApproachStates[t.id];
        float bbox_jitter = t.bbox_jitter;
        bool trend_ready = t.bbox_history.size() >= 4;
        bool history_advanced = t.bbox_history.size() != approachState.lastHistorySize;
        bool jitter_freeze = bbox_jitter >= kApproachJitterFreezeThreshold && !near_ok;

        area_trend_ratio = computeTrackAreaTrendRatio(t);
        if (trend_ready && history_advanced && !jitter_freeze) {
            if (area_trend_ratio > config.approachRatioPos) {
                approachState.positiveHits = std::min(approachState.positiveHits + 1,
                                                      kApproachPositiveFramesRequired + 1);
                if (approachState.negativeHits > 0) {
                    approachState.negativeHits--;
                }
            } else if (area_trend_ratio < config.approachRatioNeg) {
                approachState.negativeHits = std::min(approachState.negativeHits + 1,
                                                      kApproachNegativeFramesRequired + 1);
                if (approachState.positiveHits > 0) {
                    approachState.positiveHits--;
                }
            } else {
                approachState.positiveHits = std::max(0, approachState.positiveHits - 1);
                approachState.negativeHits = std::max(0, approachState.negativeHits - 1);
            }
        } else if (history_advanced && jitter_freeze) {
            approachState.positiveHits = std::max(0, approachState.positiveHits - 1);
            approachState.negativeHits = std::max(0, approachState.negativeHits - 1);
        }

        if (approachState.positiveHits >= kApproachPositiveFramesRequired) {
            approachState.isApproaching = true;
        }
        int negative_required = near_ok ? (kApproachNegativeFramesRequired + 1)
                                        : kApproachNegativeFramesRequired;
        if (approachState.negativeHits >= negative_required) {
            approachState.isApproaching = false;
        }

        approachState.lastTrend = area_trend_ratio;
        approachState.lastJitter = bbox_jitter;
        approachState.lastAreaRatio = area_ratio;
        approachState.lastHistorySize = t.bbox_history.size();
        t.is_approaching = approachState.isApproaching;
        approach_ok = t.is_approaching || near_ok || !config.requireApproach;
        bool moving_away = trend_ready &&
                           !jitter_freeze &&
                           area_trend_ratio < config.approachRatioNeg &&
                           approachState.negativeHits >= 2 &&
                           !near_ok;
        if (t.has_captured) {
            logTrackReject("gate", t.id, "already_captured", "track already captured");
            continue;
        }
        if (bbox_jitter > kApproachJitterRejectThreshold && !near_ok) {
            char detail[224];
            std::snprintf(detail, sizeof(detail), "jitter=%.3f trend=%.3f area=%.4f near=%d",
                          bbox_jitter,
                          area_trend_ratio,
                          area_ratio,
                          near_ok ? 1 : 0);
            logTrackReject("gate", t.id, "bbox_jitter", detail);
            continue;
        }
        if (!approach_ok) {
            char detail[224];
            std::snprintf(detail, sizeof(detail),
                          "approaching=%d near=%d trend=%.3f jitter=%.3f pos=%d neg=%d area=%.4f",
                          t.is_approaching ? 1 : 0,
                          near_ok ? 1 : 0,
                          area_trend_ratio,
                          bbox_jitter,
                          approachState.positiveHits,
                          approachState.negativeHits,
                          area_ratio);
            logTrackReject("gate", t.id, "not_approaching", detail);
            continue;
        }
        if (moving_away) {
            char detail[192];
            std::snprintf(detail, sizeof(detail), "trend=%.3f jitter=%.3f neg=%d area=%.4f",
                          area_trend_ratio,
                          bbox_jitter,
                          approachState.negativeHits,
                          area_ratio);
            logTrackReject("gate", t.id, "moving_away", detail);
            continue;
        }
        if (area_ratio <= config.minAreaRatio) {
            char detail[192];
            std::snprintf(detail, sizeof(detail), "area=%.4f min=%.4f near=%.4f",
                          area_ratio,
                          config.minAreaRatio,
                          config.nearAreaRatio);
            logTrackReject("gate", t.id, "too_far", detail);
            continue;
        }
        if (motion_ratio > adaptiveThresholds.maxMotionRejectRatio) {
            char detail[224];
            std::snprintf(detail, sizeof(detail), "motion=%.4f max=%.4f area=%.4f bright=%.1f ll=%.2f",
                          motion_ratio,
                          adaptiveThresholds.maxMotionRejectRatio,
                          area_ratio,
                          sceneBrightness,
                          adaptiveThresholds.lowLightStrength);
            logTrackReject("gate", t.id, "motion_large", detail);
            continue;
        }

        float person_occlusion = 0.0f;
        auto occ_it = trackOcclusionRatio.find(t.id);
        if (occ_it != trackOcclusionRatio.end()) {
            person_occlusion = occ_it->second;
        }

        CandidateEvalJob job;
        job.trackId = t.id;
        job.personRoi = person_roi.clone();
        job.fusionHistory = std::move(fusion_history);
        job.areaRatio = area_ratio;
        job.personOcclusion = person_occlusion;
        job.motionRatio = motion_ratio;
        if (enqueueCandidateEvaluation(std::move(job))) {
            clearTrackReject("gate", t.id);
        }
    }

    if (track_count > 0) {
        candidateRoundRobinOffset = (candidateRoundRobinOffset + 1) % track_count;
    }

    for (auto it = lastTrackCenters.begin(); it != lastTrackCenters.end(); ) {
        if (activeTrackIds.find(it->first) == activeTrackIds.end()) {
            it = lastTrackCenters.erase(it);
        } else {
            ++it;
        }
    }

    for (auto it = trackApproachStates.begin(); it != trackApproachStates.end(); ) {
        if (activeTrackIds.find(it->first) == activeTrackIds.end()) {
            it = trackApproachStates.erase(it);
        } else {
            ++it;
        }
    }

    for (auto it = trackPersonRoiHistory.begin(); it != trackPersonRoiHistory.end(); ) {
        if (activeTrackIds.find(it->first) == activeTrackIds.end()) {
            it = trackPersonRoiHistory.erase(it);
        } else {
            ++it;
        }
    }

    {
        std::lock_guard<std::mutex> lock(candidateEvalMutex);
        for (auto it = pendingCandidateEvalByTrack.begin(); it != pendingCandidateEvalByTrack.end(); ) {
            if (activeTrackIds.find(it->first) == activeTrackIds.end() && it->second <= 0) {
                it = pendingCandidateEvalByTrack.erase(it);
            } else {
                ++it;
            }
        }
    }

    for (auto it = reportedPersonIds.begin(); it != reportedPersonIds.end(); ) {
        if (activeTrackIds.find(*it) == activeTrackIds.end()) {
            it = reportedPersonIds.erase(it);
        } else {
            ++it;
        }
    }

    bool hasPersons = !activeTrackIds.empty();
    if (hadPersonsInScene && !hasPersons) {
        if (personEventCallback) {
            personEventCallback(-1, "all_person_left");
        }
    }
    hadPersonsInScene = hasPersons;
}
