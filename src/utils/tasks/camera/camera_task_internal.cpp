#include "camera_task_internal.h"

namespace camera_task_internal {

IrCutMode g_irCutMode = IrCutMode::Unknown;
bool g_irCutGpioReady = false;
std::chrono::steady_clock::time_point g_lastIrCutSwitchTime =
    std::chrono::steady_clock::now() - std::chrono::seconds(3600);

namespace {

cv::Mat g_gammaLut;
double g_gammaLutGamma = -1.0;

void buildGammaLut(double gamma) {
    if (std::fabs(gamma - g_gammaLutGamma) < 1e-6 && !g_gammaLut.empty()) {
        return;
    }

    cv::Mat lut(1, 256, CV_8UC1);
    uchar* values = lut.ptr();
    for (int i = 0; i < 256; ++i) {
        values[i] = cv::saturate_cast<uchar>(std::pow(i / 255.0, gamma) * 255.0);
    }

    g_gammaLut = lut;
    g_gammaLutGamma = gamma;
}

}  // namespace

float clampUnit(float value) {
    return std::max(0.0f, std::min(1.0f, value));
}

float lerpFloat(float a, float b, float t) {
    return a + (b - a) * t;
}

double lerpDouble(double a, double b, float t) {
    return a + (b - a) * static_cast<double>(t);
}

AdaptiveCaptureThresholds buildAdaptiveCaptureThresholds(
    const DeviceConfig::CaptureDefaults& config,
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

    double lowLightStart = std::max(
        kLowLightBrightnessThreshold,
        kLowLightBrightnessFloor + 1.0);
    double lowLightFloor = std::min(
        kLowLightBrightnessFloor,
        lowLightStart - 1.0);
    double range = std::max(8.0, lowLightStart - lowLightFloor);

    float lowLightStrength = 0.0f;
    if (sceneBrightness < lowLightStart) {
        lowLightStrength = clampUnit(
            static_cast<float>((lowLightStart - sceneBrightness) / range));
    }

    thresholds.lowLightStrength = lowLightStrength;
    thresholds.minClarity = lerpDouble(
        config.minClarity,
        config.minClarity * kLowLightMinClarityScale,
        lowLightStrength);
    thresholds.fallbackMinClarity = lerpDouble(
        config.fallbackMinClarity,
        config.fallbackMinClarity * kLowLightFallbackMinClarityScale,
        lowLightStrength);
    thresholds.maxMotionRatio = std::max(
        0.0015f,
        lerpFloat(
            config.maxMotionRatio,
            config.maxMotionRatio * kLowLightMotionRatioScale,
            lowLightStrength));
    thresholds.maxMotionRejectRatio = std::max(
        thresholds.maxMotionRatio + 0.003f,
        lerpFloat(
            config.maxMotionRejectRatio,
            config.maxMotionRejectRatio * kLowLightMotionRejectRatioScale,
            lowLightStrength));
    thresholds.maxBlurSeverity = std::max(
        0.12f,
        lerpFloat(
            config.maxBlurSeverity,
            config.maxBlurSeverity * kLowLightMaxBlurSeverityScale,
            lowLightStrength));
    thresholds.fallbackMaxBlurSeverity = std::max(
        thresholds.maxBlurSeverity + 0.04f,
        lerpFloat(
            config.fallbackMaxBlurSeverity,
            config.fallbackMaxBlurSeverity * kLowLightFallbackMaxBlurSeverityScale,
            lowLightStrength));
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
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        now - g_lastIrCutSwitchTime).count();
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

bool applyBrightnessBoost(
    cv::Mat& frame,
    double currentBrightness,
    const DeviceConfig::BrightnessBoostConfig& config) {
    if (currentBrightness >= config.boostThreshold ||
        currentBrightness < config.boostMinFloor) {
        return false;
    }

    double effectiveTarget = config.target;
    if (currentBrightness < 55.0) {
        effectiveTarget =
            currentBrightness + (config.target - currentBrightness) * config.darkBlend;
    }

    double gamma = config.gamma;
    if (currentBrightness < 50.0) {
        gamma = std::max(0.60, gamma - 0.10);
    }
    buildGammaLut(gamma);
    cv::LUT(frame, g_gammaLut, frame);

    double afterGamma = cv::mean(frame)[0];
    if (afterGamma < effectiveTarget && afterGamma > 1.0) {
        double residualAlpha = std::min(effectiveTarget / afterGamma, config.maxAlpha);
        double residualBeta = std::min((effectiveTarget - afterGamma) * 0.10, config.maxBeta);
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
    float areaNow = track.bbox_history.back();
    float areaPrev = track.bbox_history[track.bbox_history.size() - 1 - span];
    return (areaNow - areaPrev) / (areaPrev + 1e-6f);
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
    cv::Rect expanded(
        static_cast<int>(std::round(cx - width * 0.5f)),
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
    cv::Scalar meanVal;
    cv::Scalar stddevVal;
    cv::meanStdDev(lap, meanVal, stddevVal);
    return stddevVal[0] * stddevVal[0];
}

MultiFrameFusionResult fuseTrackHistoryPersonRoi(
    const cv::Mat& reference,
    const std::vector<cv::Mat>& history,
    const cv::Rect& focusBox,
    float lowLightStrength,
    float motionRatio) {
    MultiFrameFusionResult result;
    if (reference.empty() || history.empty() ||
        focusBox.width < 16 || focusBox.height < 16) {
        return result;
    }

    cv::Rect safeFocus = expandRectFromCenter(focusBox, 1.45f, reference.size());
    cv::Mat referenceGray;
    cv::cvtColor(reference, referenceGray, cv::COLOR_BGR2GRAY);
    result.referenceFocus = computeGrayFocusVariance(referenceGray(safeFocus));
    result.bestAlignedFocus = result.referenceFocus;

    cv::Mat referenceFocusFloat;
    referenceGray(safeFocus).convertTo(referenceFocusFloat, CV_32F);
    cv::GaussianBlur(referenceFocusFloat, referenceFocusFloat, cv::Size(0, 0), 0.8);

    float blurSigma = 1.10f + lowLightStrength * 0.90f;
    cv::Mat referenceBase;
    cv::GaussianBlur(reference, referenceBase, cv::Size(0, 0), blurSigma);

    cv::Mat accum;
    referenceBase.convertTo(accum, CV_32FC3);
    accum *= 1.28f;
    cv::Mat weightSum(reference.size(), CV_32F, cv::Scalar(1.28f));

    cv::Mat sharpest = reference.clone();
    float similaritySum = 0.0f;
    int similarityCount = 0;

    size_t historyBegin = history.size() > kMultiFrameFusionHistorySize
        ? history.size() - kMultiFrameFusionHistorySize
        : 0;
    for (size_t i = historyBegin; i < history.size(); ++i) {
        if (history[i].empty()) {
            continue;
        }

        cv::Mat candidate = history[i];
        if (candidate.size() != reference.size()) {
            cv::resize(candidate, candidate, reference.size(), 0, 0, cv::INTER_LINEAR);
        }

        cv::Mat candidateGray;
        cv::cvtColor(candidate, candidateGray, cv::COLOR_BGR2GRAY);
        cv::Mat candidateFocusFloat;
        candidateGray(safeFocus).convertTo(candidateFocusFloat, CV_32F);
        cv::GaussianBlur(candidateFocusFloat, candidateFocusFloat, cv::Size(0, 0), 0.8);

        cv::Point2d shift = cv::phaseCorrelate(referenceFocusFloat, candidateFocusFloat);
        float shiftNorm = std::sqrt(static_cast<float>(shift.x * shift.x + shift.y * shift.y));
        float maxShift = std::max(
            4.0f,
            std::min(safeFocus.width, safeFocus.height) * kMultiFrameFusionMaxShiftRatio);
        if (shiftNorm > maxShift) {
            continue;
        }

        cv::Mat transform = (cv::Mat_<double>(2, 3) << 1.0, 0.0, shift.x,
                                                       0.0, 1.0, shift.y);
        cv::Mat aligned;
        cv::warpAffine(
            candidate,
            aligned,
            transform,
            reference.size(),
            cv::INTER_LINEAR,
            cv::BORDER_REPLICATE);

        cv::Mat alignedGray;
        cv::cvtColor(aligned, alignedGray, cv::COLOR_BGR2GRAY);
        cv::Mat diff;
        cv::absdiff(alignedGray, referenceGray, diff);
        cv::Mat diffFloat;
        diff.convertTo(diffFloat, CV_32F);
        cv::Mat similarity = 1.0f - (diffFloat - 14.0f) / 62.0f;
        cv::max(similarity, 0.0f, similarity);
        cv::min(similarity, 1.0f, similarity);
        cv::GaussianBlur(similarity, similarity, cv::Size(0, 0), 1.2);

        float meanSimilarity = static_cast<float>(cv::mean(similarity(safeFocus))[0]);
        if (meanSimilarity < kMultiFrameFusionMinSimilarity) {
            continue;
        }

        cv::Mat alignedBase;
        cv::GaussianBlur(aligned, alignedBase, cv::Size(0, 0), blurSigma);
        cv::Mat alignedBaseFloat;
        alignedBase.convertTo(alignedBaseFloat, CV_32FC3);

        float frameWeightScale = 0.40f + lowLightStrength * 0.18f;
        if (motionRatio > 0.010f) {
            frameWeightScale *= 0.92f;
        }
        cv::Mat frameWeight = similarity * frameWeightScale;
        std::vector<cv::Mat> weightChannels(3, frameWeight);
        cv::Mat frameWeight3;
        cv::merge(weightChannels, frameWeight3);
        accum += alignedBaseFloat.mul(frameWeight3);
        weightSum += frameWeight;

        double alignedFocus = computeGrayFocusVariance(alignedGray(safeFocus));
        if (alignedFocus > result.bestAlignedFocus * 1.02) {
            sharpest = aligned;
            result.bestAlignedFocus = alignedFocus;
        }

        similaritySum += meanSimilarity;
        similarityCount++;
        result.acceptedFrames++;
    }

    if (result.acceptedFrames <= 0) {
        result.acceptedFrames = 1;
        result.meanSimilarity = 0.0f;
        return result;
    }

    result.meanSimilarity = similarityCount > 0
        ? (similaritySum / similarityCount)
        : 0.0f;

    cv::Mat safeWeight = weightSum + 1e-4f;
    std::vector<cv::Mat> safeWeightChannels(3, safeWeight);
    cv::Mat safeWeight3;
    cv::merge(safeWeightChannels, safeWeight3);
    cv::Mat fusedBase = accum / safeWeight3;

    cv::Mat sharpestBase;
    cv::GaussianBlur(sharpest, sharpestBase, cv::Size(0, 0), blurSigma);
    cv::Mat sharpestFloat;
    sharpest.convertTo(sharpestFloat, CV_32FC3);
    cv::Mat sharpestBaseFloat;
    sharpestBase.convertTo(sharpestBaseFloat, CV_32FC3);
    cv::Mat detail = sharpestFloat - sharpestBaseFloat;

    cv::Mat referenceFloat;
    reference.convertTo(referenceFloat, CV_32FC3);
    float detailGain = 0.78f + lowLightStrength * 0.14f;
    float referenceMix = 0.16f + std::max(0.0f, motionRatio - 0.006f) * 10.0f;
    referenceMix = std::max(0.16f, std::min(0.34f, referenceMix));

    cv::Mat fusedFloat = fusedBase + detail * detailGain;
    fusedFloat = fusedFloat * (1.0f - referenceMix) + referenceFloat * referenceMix;
    fusedFloat.convertTo(result.fused, CV_8UC3);
    return result;
}

float rect_iou(const cv::Rect& a, const cv::Rect& b) {
    int xx1 = std::max(a.x, b.x);
    int yy1 = std::max(a.y, b.y);
    int xx2 = std::min(a.x + a.width, b.x + b.width);
    int yy2 = std::min(a.y + a.height, b.y + b.height);
    int width = std::max(0, xx2 - xx1);
    int height = std::max(0, yy2 - yy1);
    float inter = static_cast<float>(width * height);
    float uni = static_cast<float>(a.area() + b.area()) - inter;
    return inter / (uni + 1e-6f);
}

float rect_overlap_ratio_on_a(const cv::Rect& a, const cv::Rect& b) {
    int xx1 = std::max(a.x, b.x);
    int yy1 = std::max(a.y, b.y);
    int xx2 = std::min(a.x + a.width, b.x + b.width);
    int yy2 = std::min(a.y + a.height, b.y + b.height);
    int width = std::max(0, xx2 - xx1);
    int height = std::max(0, yy2 - yy1);
    float inter = static_cast<float>(width * height);
    float aArea = static_cast<float>(std::max(1, a.area()));
    return inter / aArea;
}

void nmsDetections(std::vector<Detection>& dets, float iouThreshold) {
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
        cv::Rect a(
            static_cast<int>(dets[i].x1),
            static_cast<int>(dets[i].y1),
            static_cast<int>(dets[i].x2 - dets[i].x1),
            static_cast<int>(dets[i].y2 - dets[i].y1));

        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j]) {
                continue;
            }

            cv::Rect b(
                static_cast<int>(dets[j].x1),
                static_cast<int>(dets[j].y1),
                static_cast<int>(dets[j].x2 - dets[j].x1),
                static_cast<int>(dets[j].y2 - dets[j].y1));
            if (rect_iou(a, b) > iouThreshold) {
                suppressed[j] = true;
            }
        }
    }

    dets.swap(kept);
}

}  // namespace camera_task_internal
