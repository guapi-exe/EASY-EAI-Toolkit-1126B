#include "camera_task_internal.h"

using namespace camera_task_internal;

CameraTask::CameraTask(const std::string& personModel, const std::string& faceModel, int index)
    : personModelPath(personModel),
      faceModelPath(faceModel),
      cameraIndex(index),
      running(false) {
    startTime = std::chrono::steady_clock::now();
    lastFPSUpdate = startTime;

    rga_init();
}

CameraTask::~CameraTask() {
    stop();
    rga_unInit();
}

void CameraTask::start() {
    if (running) {
        return;
    }

    running = true;
    log_info("CameraTask: launching worker thread...");
    worker = std::thread(&CameraTask::run, this);
}

void CameraTask::stop() {
    if (!running && !cameraOpened && !worker.joinable()) {
        return;
    }

    log_info("CameraTask: stopping...");
    running = false;
    frameCv.notify_all();
    {
        std::lock_guard<std::mutex> lock(candidateEvalMutex);
        candidateEvalQueue.clear();
        pendingCandidateEvalByTrack.clear();
    }
    candidateEvalCv.notify_all();

    if (cameraOpened.exchange(false)) {
        mipicamera_exit(cameraIndex);
    }

    if (worker.joinable()) {
        worker.join();
    }
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
    set_max_frame_candidates(
        static_cast<size_t>(std::max(1, config.captureDefaults.maxFrameCandidates)));
}

DeviceConfig::CaptureDefaults CameraTask::getCaptureConfigSnapshot() const {
    std::lock_guard<std::mutex> lock(configMutex);
    return captureConfig;
}

DeviceConfig::BrightnessBoostConfig CameraTask::getBrightnessBoostConfigSnapshot() const {
    std::lock_guard<std::mutex> lock(configMutex);
    return brightnessBoostConfig;
}

void CameraTask::logTrackReject(
    const char* stage,
    int trackId,
    const char* reason,
    const std::string& detail) {
    auto now = std::chrono::steady_clock::now();
    std::string key = std::string(stage) + ":" + std::to_string(trackId);

    std::lock_guard<std::mutex> lock(rejectLogMutex);
    auto& state = rejectLogStates[key];

    bool hasPrevLog = state.lastLogTime.time_since_epoch().count() != 0;
    bool sameReason = state.reason == reason;
    bool withinThrottle =
        hasPrevLog &&
        std::chrono::duration_cast<std::chrono::milliseconds>(now - state.lastLogTime).count() <
            kRejectLogThrottleMs;

    if (sameReason && withinThrottle) {
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

    log_debug(
        "Track %d %s reject[%s]: %s%s",
        trackId,
        stage,
        reason,
        detail.c_str(),
        suffix);
}

void CameraTask::clearTrackReject(const char* stage, int trackId) {
    std::string key = std::string(stage) + ":" + std::to_string(trackId);
    std::lock_guard<std::mutex> lock(rejectLogMutex);
    rejectLogStates.erase(key);
}

double CameraTask::computeFocusMeasure(const cv::Mat& img) {
    if (img.empty() || img.cols <= 0 || img.rows <= 0) {
        return 0.0;
    }

    DeviceConfig::CaptureDefaults config = getCaptureConfigSnapshot();
    int scaleFactor = std::max(1, config.focusScaleFactor);

    int newWidth = img.cols / scaleFactor;
    int newHeight = img.rows / scaleFactor;
    if (newWidth <= 0 || newHeight <= 0) {
        return 0.0;
    }

    cv::Mat small;
    cv::Mat gray;
    cv::Mat lap;
    cv::resize(img, small, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
    cv::Laplacian(gray, lap, CV_64F);
    cv::Scalar meanVal;
    cv::Scalar stddevVal;
    cv::meanStdDev(lap, meanVal, stddevVal);
    return stddevVal.val[0] * stddevVal.val[0];
}

float CameraTask::computeMotionBlurSeverity(const cv::Mat& img) {
    if (img.empty() || img.cols < 16 || img.rows < 16) {
        return 1.0f;
    }

    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = img;
    }

    if (std::max(gray.cols, gray.rows) > 200) {
        float scale = 200.0f / std::max(gray.cols, gray.rows);
        cv::resize(gray, gray, cv::Size(), scale, scale, cv::INTER_AREA);
    }

    cv::Mat gx;
    cv::Mat gy;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);

    cv::Scalar gxMean;
    cv::Scalar gxStd;
    cv::Scalar gyMean;
    cv::Scalar gyStd;
    cv::meanStdDev(gx, gxMean, gxStd);
    cv::meanStdDev(gy, gyMean, gyStd);

    double energyX = gxStd.val[0] * gxStd.val[0];
    double energyY = gyStd.val[0] * gyStd.val[0];
    double totalEnergy = energyX + energyY;
    if (totalEnergy < 1.0) {
        return 1.0f;
    }

    double dirRatio = std::min(energyX, energyY) / (std::max(energyX, energyY) + 1e-6);
    float dirSeverity =
        std::max(0.0f, std::min(1.0f, static_cast<float>(1.0 - dirRatio)));

    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_32F);
    cv::Scalar lapMeanVal;
    cv::Scalar lapStdVal;
    cv::meanStdDev(lap, lapMeanVal, lapStdVal);
    double lapVar = lapStdVal.val[0] * lapStdVal.val[0];
    float sharpnessSeverity = std::max(
        0.0f,
        std::min(1.0f, static_cast<float>((200.0 - lapVar) / 200.0)));

    double angle = std::atan2(gyMean.val[0], gxMean.val[0]);
    int dx = static_cast<int>(std::round(std::cos(angle) * 2.0));
    int dy = static_cast<int>(std::round(std::sin(angle) * 2.0));
    if (dx == 0 && dy == 0) {
        dx = 1;
    }

    cv::Mat shifted;
    cv::Mat transform = (cv::Mat_<double>(2, 3) << 1, 0, dx, 0, 1, dy);
    cv::warpAffine(gray, shifted, transform, gray.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    cv::Mat diff;
    cv::absdiff(gray, shifted, diff);
    double shiftDiff = cv::mean(diff).val[0];
    float trailSeverity = std::max(
        0.0f,
        std::min(1.0f, static_cast<float>((16.0 - shiftDiff) / 16.0)));

    float severity =
        dirSeverity * 0.25f +
        sharpnessSeverity * 0.45f +
        trailSeverity * 0.30f;
    return std::max(0.0f, std::min(1.0f, severity));
}

bool CameraTask::isFrontalFace(const std::vector<cv::Point2f>& landmarks) {
    if (landmarks.size() != 5) {
        return false;
    }

    DeviceConfig::CaptureDefaults config = getCaptureConfigSnapshot();
    cv::Point2f leftEye = landmarks[0];
    cv::Point2f rightEye = landmarks[1];
    cv::Point2f nose = landmarks[2];
    cv::Point2f leftMouth = landmarks[3];
    cv::Point2f rightMouth = landmarks[4];

    float dx = rightEye.x - leftEye.x;
    float dy = rightEye.y - leftEye.y;
    if (std::fabs(dx) < 1e-5f) {
        return false;
    }

    float roll = std::atan2(dy, dx) * 180.0f / CV_PI;
    float eyeCenterX = (leftEye.x + rightEye.x) / 2.0f;
    float yaw = (nose.x - eyeCenterX) / dx;
    cv::Point2f mouthCenter(
        (leftMouth.x + rightMouth.x) * 0.5f,
        (leftMouth.y + rightMouth.y) * 0.5f);
    float eyeDistance = std::max(1e-5f, std::fabs(dx));
    float mouthOffset = std::fabs(mouthCenter.x - eyeCenterX) / eyeDistance;
    float mouthRoll = std::fabs(leftMouth.y - rightMouth.y) / eyeDistance;
    float noseVerticalRatio = (nose.y - (leftEye.y + rightEye.y) * 0.5f) / eyeDistance;
    float mouthVerticalRatio = (mouthCenter.y - nose.y) / eyeDistance;

    return (std::fabs(roll) < config.strongFrontalMaxRoll) &&
           (std::fabs(yaw) < config.strongFrontalMaxYaw) &&
           (mouthOffset < 0.24f) &&
           (mouthRoll < 0.20f) &&
           (noseVerticalRatio > 0.05f) &&
           (mouthVerticalRatio > 0.08f);
}

bool CameraTask::isSideFace(const std::vector<cv::Point2f>& landmarks) {
    if (landmarks.size() != 5) {
        return false;
    }

    cv::Point2f leftEye = landmarks[0];
    cv::Point2f rightEye = landmarks[1];
    cv::Point2f nose = landmarks[2];
    float dx = rightEye.x - leftEye.x;
    float eyeCenterX = (leftEye.x + rightEye.x) / 2.0f;
    float yaw = (nose.x - eyeCenterX) / dx;
    return (std::fabs(yaw) >= 0.25f && std::fabs(yaw) < 0.6f);
}

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
            log_info(
                "Algo FPS: total_processed=%ld, fps=%.2f, uptime=%lds",
                currentFrames,
                currentFPS.load(),
                totalElapsed.count());
        }
    }
}

bool CameraTask::enqueueCandidateEvaluation(CandidateEvalJob job) {
    DeviceConfig::CaptureDefaults config = getCaptureConfigSnapshot();
    std::lock_guard<std::mutex> lock(candidateEvalMutex);
    int pendingForTrack = pendingCandidateEvalByTrack[job.trackId];
    if (pendingForTrack >= config.candidatePerTrackMaxPending) {
        char detail[192];
        std::snprintf(
            detail,
            sizeof(detail),
            "pending=%d max=%d queue=%zu",
            pendingForTrack,
            config.candidatePerTrackMaxPending,
            candidateEvalQueue.size());
        logTrackReject("queue", job.trackId, "pending_limit", detail);
        return false;
    }

    auto removePendingForTrack = [this](int trackId) {
        auto droppedIt = pendingCandidateEvalByTrack.find(trackId);
        if (droppedIt != pendingCandidateEvalByTrack.end()) {
            droppedIt->second--;
            if (droppedIt->second <= 0) {
                pendingCandidateEvalByTrack.erase(droppedIt);
            }
        }
    };

    auto dropJobAt = [&](size_t index) {
        int droppedTrackId = candidateEvalQueue[index].trackId;
        candidateEvalQueue.erase(candidateEvalQueue.begin() + index);
        removePendingForTrack(droppedTrackId);
    };

    if (candidateEvalQueue.size() >= static_cast<size_t>(config.candidateQueueMax)) {
        size_t dropIndex = candidateEvalQueue.size();

        if (pendingForTrack > 0) {
            for (size_t i = 0; i < candidateEvalQueue.size(); ++i) {
                if (candidateEvalQueue[i].trackId == job.trackId) {
                    dropIndex = i;
                    break;
                }
            }
        }

        if (dropIndex == candidateEvalQueue.size()) {
            for (size_t i = 0; i < candidateEvalQueue.size(); ++i) {
                int queuedTrackId = candidateEvalQueue[i].trackId;
                auto queuedIt = pendingCandidateEvalByTrack.find(queuedTrackId);
                if (queuedIt != pendingCandidateEvalByTrack.end() && queuedIt->second > 1) {
                    dropIndex = i;
                    break;
                }
            }
        }

        if (dropIndex == candidateEvalQueue.size()) {
            for (size_t i = 0; i < candidateEvalQueue.size(); ++i) {
                if (candidateEvalQueue[i].trackId != job.trackId) {
                    dropIndex = i;
                    break;
                }
            }
        }

        if (dropIndex == candidateEvalQueue.size()) {
            char detail[192];
            std::snprintf(
                detail,
                sizeof(detail),
                "queue=%zu pending=%d no_replace_slot",
                candidateEvalQueue.size(),
                pendingForTrack);
            logTrackReject("queue", job.trackId, "queue_full", detail);
            return false;
        }

        dropJobAt(dropIndex);
    }

    pendingCandidateEvalByTrack[job.trackId] = pendingForTrack + 1;
    candidateEvalQueue.push_back(std::move(job));
    clearTrackReject("queue", candidateEvalQueue.back().trackId);
    candidateEvalCv.notify_one();
    return true;
}
