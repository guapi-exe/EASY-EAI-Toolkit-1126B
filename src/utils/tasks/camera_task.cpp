#include "camera_task.h"
#include "main.h"
#include "person_detect.h"
#include "face_detect.h"
#include "sort_tracker.h"
#include <fstream>
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
}

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
    log_info("CameraTask: stopped");
}

void CameraTask::setUploadCallback(UploadCallback cb) {
    uploadCallback = cb;
}

void CameraTask::setPersonEventCallback(PersonEventCallback cb) {
    personEventCallback = cb;
}

// -------------------- 图像清晰度计算 --------------------
double CameraTask::computeFocusMeasure(const Mat& img) {
    if (img.empty() || img.cols <= 0 || img.rows <= 0) {
        return 0.0;
    }
    
    int scale_factor = CAPTURE_FOCUS_SCALE_FACTOR;
    
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

bool CameraTask::isFrontalFace(const std::vector<cv::Point2f>& landmarks) {
    if (landmarks.size() != 5) return false;
    cv::Point2f left_eye = landmarks[0];
    cv::Point2f right_eye = landmarks[1];
    cv::Point2f nose = landmarks[2];
    // cv::Point2f left_mouth = landmarks[3];
    // cv::Point2f right_mouth = landmarks[4];

    float dx = right_eye.x - left_eye.x;
    float dy = right_eye.y - left_eye.y;
    float roll = atan2(dy, dx) * 180.0 / CV_PI;

    float eye_center_x = (left_eye.x + right_eye.x) / 2.0;
    float yaw = (nose.x - eye_center_x) / dx;
    return (fabs(roll) < 20.0) && (fabs(yaw) < 0.25);
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
    std::lock_guard<std::mutex> lock(candidateEvalMutex);
    int pending_for_track = pendingCandidateEvalByTrack[job.trackId];
    if (pending_for_track >= CAPTURE_CANDIDATE_PER_TRACK_MAX_PENDING) {
        return false;
    }
    if (candidateEvalQueue.size() >= CAPTURE_CANDIDATE_QUEUE_MAX) {
        int dropped_track_id = candidateEvalQueue.front().trackId;
        candidateEvalQueue.pop_front();
        auto dropped_it = pendingCandidateEvalByTrack.find(dropped_track_id);
        if (dropped_it != pendingCandidateEvalByTrack.end()) {
            dropped_it->second--;
            if (dropped_it->second <= 0) {
                pendingCandidateEvalByTrack.erase(dropped_it);
            }
        }
    }

    pendingCandidateEvalByTrack[job.trackId] = pending_for_track + 1;
    candidateEvalQueue.push_back(std::move(job));
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
            Mat person_roi_resized;
            int target_width = min(CAPTURE_FACE_INPUT_MAX_WIDTH, job.personRoi.cols);
            int target_height = static_cast<int>(job.personRoi.rows * target_width / (float)job.personRoi.cols);

            if (target_width > 0 && target_height > 0) {
                cv::resize(job.personRoi, person_roi_resized, Size(target_width, target_height), 0, 0, cv::INTER_LINEAR);

                std::vector<det> face_result;
                int num_faces = face_detect_run(faceCtx, person_roi_resized, face_result);
                if (num_faces > 0 && !face_result.empty()) {
                    int best_idx = 0;
                    for (int i = 1; i < num_faces; ++i) {
                        if (face_result[i].score > face_result[best_idx].score) {
                            best_idx = i;
                        }
                    }

                    float face_scale_x = (float)job.personRoi.cols / (float)person_roi_resized.cols;
                    float face_scale_y = (float)job.personRoi.rows / (float)person_roi_resized.rows;

                    det best_face = face_result[best_idx];
                    if (best_face.score >= CAPTURE_MIN_FACE_SCORE && best_face.landmarks.size() >= 3) {
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
                            if (face_short_side >= CAPTURE_MIN_FACE_BOX_SHORT_SIDE &&
                                face_area >= CAPTURE_MIN_FACE_BOX_AREA) {
                                float margin_left = static_cast<float>(base_fbox.x);
                                float margin_right = static_cast<float>(job.personRoi.cols - (base_fbox.x + base_fbox.width));
                                float margin_top = static_cast<float>(base_fbox.y);
                                float margin_bottom = static_cast<float>(job.personRoi.rows - (base_fbox.y + base_fbox.height));
                                float margin_x_ratio = std::min(margin_left, margin_right) / std::max(1, base_fbox.width);
                                float margin_y_ratio = std::min(margin_top, margin_bottom) / std::max(1, base_fbox.height);
                                float min_margin_ratio = std::min(margin_x_ratio, margin_y_ratio);

                                float face_edge_occlusion = 0.0f;
                                if (min_margin_ratio < CAPTURE_FACE_EDGE_MIN_MARGIN) {
                                    face_edge_occlusion =
                                        (CAPTURE_FACE_EDGE_MIN_MARGIN - min_margin_ratio) / CAPTURE_FACE_EDGE_MIN_MARGIN;
                                    face_edge_occlusion = std::max(0.0f, std::min(1.0f, face_edge_occlusion));
                                }

                                cv::Point2f left_eye = best_face.landmarks[0];
                                cv::Point2f right_eye = best_face.landmarks[1];
                                float dx = right_eye.x - left_eye.x;
                                if (std::fabs(dx) >= 1e-5f) {
                                    float eye_center_x = (left_eye.x + right_eye.x) / 2.0f;
                                    float yaw = std::fabs((best_face.landmarks[2].x - eye_center_x) / dx);
                                    bool frontal_ok = isFrontalFace(best_face.landmarks);
                                    double current_clarity = computeFocusMeasure(job.personRoi(base_fbox));

                                    if ((!CAPTURE_REQUIRE_FRONTAL_FACE || frontal_ok) &&
                                        current_clarity > CAPTURE_MIN_CLARITY &&
                                        yaw < CAPTURE_MAX_YAW) {
                                        int crop_w = std::max(1, static_cast<int>(base_fbox.width * CAPTURE_HEADSHOT_EXPAND_RATIO));
                                        int crop_h = std::max(1, static_cast<int>(base_fbox.height * CAPTURE_HEADSHOT_EXPAND_RATIO));
                                        int crop_cx = base_fbox.x + base_fbox.width / 2;
                                        int crop_cy = base_fbox.y + base_fbox.height / 2 + static_cast<int>(base_fbox.height * CAPTURE_HEADSHOT_DOWN_SHIFT);
                                        int crop_x = crop_cx - crop_w / 2;
                                        int crop_y = crop_cy - crop_h / 2;

                                        crop_x = std::max(0, std::min(job.personRoi.cols - crop_w, crop_x));
                                        crop_y = std::max(0, std::min(job.personRoi.rows - crop_h, crop_y));
                                        Rect fbox(crop_x, crop_y,
                                                  std::min(crop_w, job.personRoi.cols - crop_x),
                                                  std::min(crop_h, job.personRoi.rows - crop_y));

                                        if (fbox.width > 0 && fbox.height > 0) {
                                            Mat face_aligned = job.personRoi(fbox).clone();
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

                                            float area_score = 1.0f / (1.0f + std::fabs(job.areaRatio - 0.15f) / 0.15f);
                                            float person_occ_norm = job.personOcclusion / std::max(1e-6f, CAPTURE_MAX_PERSON_OCCLUSION);
                                            person_occ_norm = std::max(0.0f, std::min(1.5f, person_occ_norm));
                                            float face_edge_occ_norm = face_edge_occlusion / std::max(1e-6f, CAPTURE_MAX_FACE_EDGE_OCCLUSION);
                                            face_edge_occ_norm = std::max(0.0f, std::min(1.5f, face_edge_occ_norm));
                                            float occlusion_penalty = person_occ_norm * 0.7f + face_edge_occ_norm * 0.3f;
                                            float motion_penalty = std::min(1.25f, job.motionRatio / std::max(1e-6f, CAPTURE_MAX_MOTION_RATIO));

                                            Track::FrameData frame_data;
                                            frame_data.score = current_clarity * quality_weight +
                                                               area_score * 1000 * area_weight -
                                                               occlusion_penalty * CAPTURE_OCCLUSION_SCORE_PENALTY -
                                                               motion_penalty * CAPTURE_MOTION_SCORE_PENALTY;
                                            frame_data.person_roi = job.personRoi.clone();
                                            frame_data.face_roi = face_aligned;
                                            frame_data.has_face = true;
                                            frame_data.is_frontal = frontal_ok;
                                            frame_data.yaw_abs = yaw;
                                            frame_data.clarity = current_clarity;
                                            frame_data.area_ratio = job.areaRatio;
                                            frame_data.person_occlusion = job.personOcclusion;
                                            frame_data.face_edge_occlusion = face_edge_occlusion;
                                            frame_data.motion_ratio = job.motionRatio;
                                            add_frame_candidate(job.trackId, frame_data);
                                        }
                                    }
                                }
                            }
                        }
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

        brightnessSampleCounter++;
        if (brightnessSampleCounter % CAMERA_BRIGHTNESS_SAMPLE_INTERVAL == 0) {
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
            } else if (filteredBrightness >= CAMERA_BRIGHTNESS_WHITE_THRESHOLD) {
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
        {
            std::lock_guard<std::mutex> lock(frameMutex);
            // 推理线程尚未消费上一帧时，直接丢弃当前帧，避免无效4K拷贝。
            if (latestFrameSeq == consumedFrameSeq) {
                latestFrame = frame.clone();
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
            log_info("Capture FPS: fps=%.2f, brightness=%.1f(raw=%.1f), ircut=%s, black_th=%.1f",
                     captureFps,
                     environmentBrightness.load(),
                     lastBrightnessRaw,
                     irCutModeToString(g_irCutMode),
                     brightnessBlackThreshold.load());
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

    bool runPersonDetect = (personDetectCounter % CAPTURE_PERSON_DETECT_INTERVAL == 0) || cachedTracks.empty();
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
        cv::Rect box((int)tr.bbox.x, (int)tr.bbox.y, (int)tr.bbox.width, (int)tr.bbox.height);
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

    for (auto& t : tracks) {
        activeTrackIds.insert(t.id);

        Rect bbox_720p((int)t.bbox.x, (int)t.bbox.y, (int)t.bbox.width, (int)t.bbox.height);
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

        if (t.hits < CAPTURE_MIN_TRACK_HITS) {
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
        
        Rect bbox_4k(orig_x, orig_y, orig_width, orig_height);

        float current_area_4k = bbox_4k.width * bbox_4k.height;
        float area_ratio = current_area_4k / (CAMERA_WIDTH * CAMERA_HEIGHT);

        float area_trend_ratio = 0.0f;
        if (t.bbox_history.size() >= 5) {
            float area_now = t.bbox_history.back();
            float area_prev = t.bbox_history[t.bbox_history.size() - 5];
            area_trend_ratio = (area_now - area_prev) / (area_prev + 1e-6f);

            // 先更新接近状态，再用于后续抓拍门控
            if (area_trend_ratio > CAPTURE_APPROACH_RATIO_POS) {
                t.is_approaching = true;
            } else if (area_trend_ratio < CAPTURE_APPROACH_RATIO_NEG) {
                t.is_approaching = false;
            }
        }

        bool near_ok = area_ratio >= CAPTURE_NEAR_AREA_RATIO;
        bool approach_ok = t.is_approaching || near_ok || (CAPTURE_REQUIRE_APPROACH == 0);
        if (t.has_captured || !approach_ok) {
            continue;
        }
        if (area_trend_ratio < CAPTURE_APPROACH_RATIO_NEG && !near_ok) {
            continue;
        }
        if (area_ratio <= CAPTURE_MIN_AREA_RATIO || motion_ratio > CAPTURE_MAX_MOTION_REJECT_RATIO) {
            continue;
        }

        float person_occlusion = 0.0f;
        auto occ_it = trackOcclusionRatio.find(t.id);
        if (occ_it != trackOcclusionRatio.end()) {
            person_occlusion = occ_it->second;
        }

        Mat person_roi = frame(bbox_4k);

        if (person_roi.empty() || person_roi.cols <= 0 || person_roi.rows <= 0) {
            continue;
        }
        CandidateEvalJob job;
        job.trackId = t.id;
        job.personRoi = person_roi.clone();
        job.areaRatio = area_ratio;
        job.personOcclusion = person_occlusion;
        job.motionRatio = motion_ratio;
        enqueueCandidateEvaluation(std::move(job));
    }

    for (auto it = lastTrackCenters.begin(); it != lastTrackCenters.end(); ) {
        if (activeTrackIds.find(it->first) == activeTrackIds.end()) {
            it = lastTrackCenters.erase(it);
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
