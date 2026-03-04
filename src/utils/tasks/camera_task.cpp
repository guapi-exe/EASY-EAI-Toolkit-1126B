#include "camera_task.h"
#include "main.h"
#include "person_detect.h"
#include "face_detect.h"
#include "sort_tracker.h"
extern "C" {
#include "log.h"
#include "camera.h"
#include "rga_wrapper.h"
}

using namespace cv;
using namespace std;

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
    if (!running) return;
    log_info("CameraTask: stopping...");
    running = false;
    if (worker.joinable()) worker.join();
    log_info("CameraTask: stopped");
}

void CameraTask::setUploadCallback(UploadCallback cb) {
    uploadCallback = cb;
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
            log_info("帧数统计: 总帧数=%ld, 当前FPS=%.2f, 运行时间=%lds", currentFrames, currentFPS.load(), totalElapsed.count());
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

    log_info("CameraTask: starting main processing loop...");
    vector<unsigned char> buffer(IMAGE_SIZE);
    while (running) {
        if (mipicamera_getframe(cameraIndex, reinterpret_cast<char*>(buffer.data())) != 0) continue;
        Mat frame(CAMERA_HEIGHT, CAMERA_WIDTH, CV_8UC3, buffer.data());
        if (frame.empty() || frame.cols <= 0 || frame.rows <= 0) {
            log_error("CameraTask: invalid frame dimensions (width=%d, height=%d)", frame.cols, frame.rows);
            continue;
        }
        
        // 帧数统计
        totalFrames++;
        updateFPS();
        
        processFrame(frame, personCtx, faceCtx);
    }

    log_info("CameraTask: main loop exited, cleaning up...");
    mipicamera_exit(cameraIndex);
    person_detect_release(personCtx);
    face_detect_release(faceCtx);
    log_info("CameraTask: cleanup completed");
}

void CameraTask::captureSnapshot() {
    std::vector<unsigned char> buffer(IMAGE_SIZE);
    if (mipicamera_getframe(cameraIndex, reinterpret_cast<char*>(buffer.data())) == 0) {
        cv::Mat frame(CAMERA_HEIGHT, CAMERA_WIDTH, CV_8UC3, buffer.data());
        if (!frame.empty()) {
            if (uploadCallback) {
                uploadCallback(frame.clone(), 0, "all");
                log_info("CameraTask: snapshot uploaded");
            }
        } else {
            log_error("CameraTask: snapshot empty frame");
        }
    } else {
        log_error("CameraTask: snapshot failed to get frame");
    }
}


void CameraTask::processFrame(const Mat& frame, rknn_context personCtx, rknn_context faceCtx) {
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

    detect_result_group_t detect_result_group;
    person_detect_run(personCtx, resized_frame, &detect_result_group);

    vector<Detection> dets;
    for (int i=0; i<detect_result_group.count; i++) {
        detect_result_t& d = detect_result_group.results[i];
        if (d.prop < 0.7) continue;
        
        Rect roi_720p(max(0, d.box.left), max(0, d.box.top),
                      min(IMAGE_WIDTH-1, d.box.right) - max(0, d.box.left),
                      min(IMAGE_HEIGHT-1, d.box.bottom) - max(0, d.box.top));
        if (roi_720p.width <=0 || roi_720p.height <=0) continue;
        
        Detection det; 
        det.roi = resized_frame(roi_720p); 
        det.x1 = roi_720p.x; 
        det.y1 = roi_720p.y; 
        det.x2 = roi_720p.x + roi_720p.width; 
        det.y2 = roi_720p.y + roi_720p.height; 
        det.prop = d.prop;
        dets.push_back(det);
    }

    vector<Track> tracks = sort_update(dets);
    std::unordered_set<int> activeTrackIds;

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

        bool approach_ok = t.is_approaching || (CAPTURE_REQUIRE_APPROACH == 0);
        if (t.has_captured || !approach_ok) {
            continue;
        }
        if (area_ratio <= CAPTURE_MIN_AREA_RATIO || motion_ratio > CAPTURE_MAX_MOTION_RATIO) {
            continue;
        }

        int& skip_counter = trackFaceDetectSkipCounters[t.id];
        skip_counter = (skip_counter + 1) % CAPTURE_FACE_DETECT_INTERVAL;
        if (skip_counter != 0) {
            continue;
        }

        Mat person_roi = frame(bbox_4k);

        if (person_roi.empty() || person_roi.cols <= 0 || person_roi.rows <= 0) {
            continue;
        }

        Mat person_roi_resized;
        int target_width = min(CAPTURE_FACE_INPUT_MAX_WIDTH, person_roi.cols);
        int target_height = static_cast<int>(person_roi.rows * target_width / (float)person_roi.cols);

        if (target_width <= 0 || target_height <= 0) {
            continue;
        }

        cv::resize(person_roi, person_roi_resized, Size(target_width, target_height), 0, 0, cv::INTER_LINEAR);

        if (t.bbox_history.size() >= 5) {
            float area_now = t.bbox_history.back();
            float area_prev = t.bbox_history[t.bbox_history.size() - 5];
            float ratio = (area_now - area_prev) / (area_prev + 1e-6f);
            
            // 判断是否正在接近摄像机
            if (ratio > CAPTURE_APPROACH_RATIO_POS) {
                t.is_approaching = true;
            } else if (ratio < CAPTURE_APPROACH_RATIO_NEG) {
                t.is_approaching = false;
            }
        }

        std::vector<det> face_result;
        int num_faces = face_detect_run(faceCtx, person_roi_resized, face_result);
        if (num_faces <= 0 || face_result.empty()) {
            continue;
        }

        int best_idx = 0;
        for (int i = 1; i < num_faces; ++i) {
            if (face_result[i].score > face_result[best_idx].score) {
                best_idx = i;
            }
        }

        float face_scale_x = (float)person_roi.cols / (float)person_roi_resized.cols;
        float face_scale_y = (float)person_roi.rows / (float)person_roi_resized.rows;

        det best_face = face_result[best_idx];
        if (best_face.score < CAPTURE_MIN_FACE_SCORE) {
            continue;
        }
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
        base_fbox.x = std::max(0, std::min(base_fbox.x, person_roi.cols - 1));
        base_fbox.y = std::max(0, std::min(base_fbox.y, person_roi.rows - 1));
        base_fbox.width = std::min(base_fbox.width, person_roi.cols - base_fbox.x);
        base_fbox.height = std::min(base_fbox.height, person_roi.rows - base_fbox.y);
        if (base_fbox.width <= 0 || base_fbox.height <= 0) {
            continue;
        }

        cv::Point2f left_eye = best_face.landmarks[0];
        cv::Point2f right_eye = best_face.landmarks[1];
        float dx = right_eye.x - left_eye.x;
        if (std::fabs(dx) < 1e-5f) {
            continue;
        }
        float eye_center_x = (left_eye.x + right_eye.x) / 2.0f;
        float yaw = std::fabs((best_face.landmarks[2].x - eye_center_x) / dx);

        double current_clarity = computeFocusMeasure(person_roi(base_fbox));
        if (current_clarity <= CAPTURE_MIN_CLARITY) {
            continue;
        }

        log_debug("Track ID=%d, clarity=%.2f, area_ratio=%.4f, yaw=%.4f", t.id, current_clarity, area_ratio, yaw);
        if (yaw >= CAPTURE_MAX_YAW) {
            continue;
        }

        int crop_w = std::max(1, static_cast<int>(base_fbox.width * CAPTURE_HEADSHOT_EXPAND_RATIO));
        int crop_h = std::max(1, static_cast<int>(base_fbox.height * CAPTURE_HEADSHOT_EXPAND_RATIO));
        int crop_cx = base_fbox.x + base_fbox.width / 2;
        int crop_cy = base_fbox.y + base_fbox.height / 2 + static_cast<int>(base_fbox.height * CAPTURE_HEADSHOT_DOWN_SHIFT);
        int crop_x = crop_cx - crop_w / 2;
        int crop_y = crop_cy - crop_h / 2;

        crop_x = std::max(0, std::min(person_roi.cols - crop_w, crop_x));
        crop_y = std::max(0, std::min(person_roi.rows - crop_h, crop_y));
        Rect fbox(crop_x, crop_y,
                  std::min(crop_w, person_roi.cols - crop_x),
                  std::min(crop_h, person_roi.rows - crop_y));
        if (fbox.width <= 0 || fbox.height <= 0) {
            continue;
        }

        Mat face_aligned = person_roi(fbox).clone();

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

        float ideal_area = CAMERA_WIDTH * CAMERA_HEIGHT * 0.15f;
        float area_score = 1.0f / (1.0f + abs(current_area_4k - ideal_area) / ideal_area);
        double current_score = current_clarity * quality_weight + area_score * 1000 * area_weight;

        Track::FrameData frame_data;
        frame_data.score = current_score;
        frame_data.person_roi = person_roi.clone();
        frame_data.face_roi = face_aligned;
        frame_data.has_face = true;
        frame_data.clarity = current_clarity;
        frame_data.area_ratio = area_ratio;
        add_frame_candidate(t.id, frame_data);
    }

    for (auto it = lastTrackCenters.begin(); it != lastTrackCenters.end(); ) {
        if (activeTrackIds.find(it->first) == activeTrackIds.end()) {
            it = lastTrackCenters.erase(it);
        } else {
            ++it;
        }
    }

    for (auto it = trackFaceDetectSkipCounters.begin(); it != trackFaceDetectSkipCounters.end(); ) {
        if (activeTrackIds.find(it->first) == activeTrackIds.end()) {
            it = trackFaceDetectSkipCounters.erase(it);
        } else {
            ++it;
        }
    }
}
