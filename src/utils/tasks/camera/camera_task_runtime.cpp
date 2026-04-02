#include "camera_task_internal.h"

using namespace camera_task_internal;

namespace {

bool fileExists(const std::string& path) {
    FILE* fp = fopen(path.c_str(), "r");
    if (!fp) {
        return false;
    }
    fclose(fp);
    return true;
}

}  // namespace

void CameraTask::run() {
    log_info("CameraTask: starting camera task...");
    log_info("CameraTask: working directory: %s", getcwd(NULL, 0));
    log_info("CameraTask: person model path: %s", personModelPath.c_str());
    log_info("CameraTask: face model path: %s", faceModelPath.c_str());

    if (fileExists(personModelPath)) {
        log_info("CameraTask: person model file exists");
    } else {
        log_error("CameraTask: person model file NOT found: %s", personModelPath.c_str());
        return;
    }

    if (fileExists(faceModelPath)) {
        log_info("CameraTask: face model file exists");
    } else {
        log_error("CameraTask: face model file NOT found: %s", faceModelPath.c_str());
        return;
    }

    log_info("CameraTask: loading person model...");
    fflush(stdout);
    fflush(stderr);

    auto startTimePoint = std::chrono::steady_clock::now();
    rknn_context personCtx;
    rknn_context faceCtx;

    int ret = person_detect_init(&personCtx, personModelPath.c_str());
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - startTimePoint);

    if (ret != 0) {
        log_error(
            "CameraTask: person_detect_init failed with code %d (model: %s, time: %lds)",
            ret,
            personModelPath.c_str(),
            elapsed.count());
        return;
    }
    log_info(
        "CameraTask: person model loaded successfully in %ld seconds",
        elapsed.count());

    log_info("CameraTask: loading face model...");
    fflush(stdout);
    fflush(stderr);

    startTimePoint = std::chrono::steady_clock::now();
    ret = face_detect_init(&faceCtx, faceModelPath.c_str());
    elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - startTimePoint);

    if (ret != 0) {
        log_error(
            "CameraTask: face_detect_init failed with code %d (model: %s, time: %lds)",
            ret,
            faceModelPath.c_str(),
            elapsed.count());
        person_detect_release(personCtx);
        return;
    }
    log_info(
        "CameraTask: face model loaded successfully in %ld seconds",
        elapsed.count());

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

    log_info(
        "CameraTask: initializing camera (index=%d, resolution=%dx%d)...",
        cameraIndex,
        CAMERA_WIDTH,
        CAMERA_HEIGHT);

    if (mipicamera_init(cameraIndex, CAMERA_WIDTH, CAMERA_HEIGHT, 0) != 0) {
        log_error("CameraTask: camera init failed (index=%d)", cameraIndex);
        person_detect_release(personCtx);
        face_detect_release(faceCtx);
        return;
    }
    cameraOpened = true;
    mipicamera_set_format(cameraIndex, CAMERA_FORMAT);
    log_info(
        "CameraTask: camera initialized successfully (format=%s)",
        CAMERA_FORMAT == RK_FORMAT_YCbCr_420_SP ? "NV12" :
        CAMERA_FORMAT == RK_FORMAT_BGR_888 ? "BGR888" :
        CAMERA_FORMAT == RK_FORMAT_RGB_888 ? "RGB888" : "UNKNOWN");

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
        cv::Mat frame;
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
            log_error(
                "CameraTask: invalid frame dimensions (width=%d, height=%d)",
                frame.cols,
                frame.rows);
            continue;
        }

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

    g_lastIrCutSwitchTime =
        std::chrono::steady_clock::now() - std::chrono::seconds(3600);
    switchIrCutWhite();

    while (running) {
        if (mipicamera_getframe(cameraIndex, reinterpret_cast<char*>(buffer.data())) != 0) {
            if (!running) {
                break;
            }
            continue;
        }

        cv::Mat frame(CAMERA_HEIGHT, CAMERA_WIDTH, CV_8UC3, buffer.data());
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
            auto sinceLastSwitch = std::chrono::duration_cast<std::chrono::seconds>(
                now - g_lastIrCutSwitchTime).count();
            double blackThreshold = brightnessBlackThreshold.load();
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

        bool published = false;
        bool boosted = false;
        {
            std::lock_guard<std::mutex> lock(frameMutex);
            if (latestFrameSeq == consumedFrameSeq) {
                latestFrame = frame.clone();
                if (filteredBrightness > boostConfig.boostMinFloor &&
                    filteredBrightness < boostConfig.boostThreshold) {
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
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - captureFpsWindowStart).count();
        if (elapsed >= 5) {
            double captureFps =
                static_cast<double>(captureFramesInWindow) / static_cast<double>(elapsed);
            log_info(
                "Capture FPS: fps=%.2f, brightness=%.1f(raw=%.1f), ircut=%s, black_th=%.1f, boost=%s",
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
