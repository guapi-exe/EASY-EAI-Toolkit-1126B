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
    if (running) return;
    running = true;
    worker = thread(&CameraTask::run, this);
}

void CameraTask::stop() {
    running = false;
    if (worker.joinable()) worker.join();
}

void CameraTask::setUploadCallback(UploadCallback cb) {
    uploadCallback = cb;
}

// -------------------- 图像清晰度计算 --------------------
double CameraTask::computeFocusMeasure(const Mat& img) {
    int scale_factor = 2;
    
    Mat small, gray, lap;
    cv::resize(img, small, Size(img.cols/scale_factor, img.rows/scale_factor), 0, 0, cv::INTER_LINEAR);
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
    cv::Point2f left_mouth = landmarks[3];
    cv::Point2f right_mouth = landmarks[4];

    float dx = right_eye.x - left_eye.x;
    float dy = right_eye.y - left_eye.y;
    float roll = atan2(dy, dx) * 180.0 / CV_PI;

    float eye_center_x = (left_eye.x + right_eye.x) / 2.0;
    float yaw = (nose.x - eye_center_x) / dx;

    float eye_center_y = (left_eye.y + right_eye.y) / 2.0;
    float mouth_center_y = (left_mouth.y + right_mouth.y) / 2.0;
    float pitch = (mouth_center_y - eye_center_y) / dx;
    return (fabs(roll) < 40.0) && (fabs(yaw) < 0.4);
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
    rknn_context personCtx, faceCtx;
    if (person_detect_init(&personCtx, personModelPath.c_str()) != 0) {
        log_debug("CameraTask: person_detect_init failed");
        return;
    }
    if (face_detect_init(&faceCtx, faceModelPath.c_str()) != 0) {
        log_debug("CameraTask: face_detect_init failed");
        person_detect_release(personCtx);
        return;
    }
    sort_init();

    set_upload_callback([this](const cv::Mat& img, int id, const std::string& type) {
        if (uploadCallback) {
            uploadCallback(img, id, type);
        }
    }, &capturedPersonIds, &capturedFaceIds);
    
    if (mipicamera_init(cameraIndex, CAMERA_WIDTH, CAMERA_HEIGHT, 0) != 0) {
        log_debug("CameraTask: Camera init failed");
        person_detect_release(personCtx);
        face_detect_release(faceCtx);
        return;
    }
    
   /*
   if (usbcamera_init(cameraIndex, CAMERA_WIDTH, CAMERA_HEIGHT, 0) != 0) {
        log_debug("CameraTask: Camera init failed");
        person_detect_release(personCtx);
        face_detect_release(faceCtx);
        return;
    }
   */
    

    vector<unsigned char> buffer(IMAGE_SIZE);
    while (running) {
        if (mipicamera_getframe(cameraIndex, reinterpret_cast<char*>(buffer.data())) != 0) continue;
        Mat frame(CAMERA_HEIGHT, CAMERA_WIDTH, CV_8UC3, buffer.data());
        if (frame.empty()) continue;
        
        // 帧数统计
        totalFrames++;
        updateFPS();
        
        processFrame(frame, personCtx, faceCtx);
    }

    mipicamera_exit(cameraIndex);
    person_detect_release(personCtx);
    face_detect_release(faceCtx);
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
    log_info("Test");
    cv::resize(frame, resized_frame, Size(IMAGE_WIDTH, IMAGE_HEIGHT), 0, 0, cv::INTER_NEAREST);
    log_info("Test");
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
        
        // 使用720p图像的ROI用于直方图计算（追踪用）
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

    for (auto& t : tracks) {
        Rect bbox_720p((int)t.bbox.x, (int)t.bbox.y, (int)t.bbox.width, (int)t.bbox.height);
        if (bbox_720p.width <=0 || bbox_720p.height <=0) continue;
        
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
        
        Mat person_roi = frame(bbox_4k).clone();
        
        if (person_roi.empty() || person_roi.cols <= 0 || person_roi.rows <= 0) {
            log_error("Invalid person_roi: empty=%d, cols=%d, rows=%d", 
                      person_roi.empty(), person_roi.cols, person_roi.rows);
            continue;
        }
        
        Mat person_roi_resized;
        int target_width = min(640, person_roi.cols);
        int target_height = static_cast<int>(person_roi.rows * target_width / (float)person_roi.cols);
        
        if (target_width <= 0 || target_height <= 0) {
            log_error("Invalid resize target: width=%d, height=%d (src: %dx%d)", 
                      target_width, target_height, person_roi.cols, person_roi.rows);
            continue;
        }
        
        log_info("Test2");
        cv::resize(person_roi, person_roi_resized, Size(target_width, target_height), 0, 0, cv::INTER_NEAREST);
        log_info("Test2");

        if (t.bbox_history.size() >= 5) {
            float area_now = t.bbox_history.back();
            float area_prev = t.bbox_history[t.bbox_history.size() - 5];
            float ratio = (area_now - area_prev) / (area_prev + 1e-6f);
            
            // 判断是否正在接近摄像机
            if (ratio > 0.1f) {
                t.is_approaching = true;
            } else if (ratio < -0.1f) {
                t.is_approaching = false;
            }
        }

        if (t.is_approaching && !t.has_captured) {
            float current_area_4k = bbox_4k.width * bbox_4k.height;
            float area_ratio = current_area_4k / (CAMERA_WIDTH * CAMERA_HEIGHT);
            if (area_ratio > 0.05f) {
                double current_clarity = computeFocusMeasure(person_roi_resized);
                if (current_clarity > 50) {
                    std::vector<det> face_result;
                    face_detect_run(faceCtx, person_roi_resized, face_result);
                    float face_scale_x = (float)person_roi.cols / (float)person_roi_resized.cols;
                    float face_scale_y = (float)person_roi.rows / (float)person_roi_resized.rows;
                    for (auto& face : face_result) {
                        face.box.x = static_cast<int>(face.box.x * face_scale_x);
                        face.box.y = static_cast<int>(face.box.y * face_scale_y);
                        face.box.width = static_cast<int>(face.box.width * face_scale_x);
                        face.box.height = static_cast<int>(face.box.height * face_scale_y);
                    }
                    if (!face_result.empty()) {
                        // 处理人脸框
                        Rect fbox = cv::Rect(face_result[0].box);
                        int w_expand = static_cast<int>(fbox.width * 1.0 / 2.0);
                        int h_expand = static_cast<int>(fbox.height * 1.0 / 2.0);
                        fbox.x = std::max(0, fbox.x - w_expand);
                        fbox.y = std::max(0, fbox.y - h_expand);
                        fbox.width = std::min(person_roi.cols - fbox.x, fbox.width + 2 * w_expand);
                        fbox.height = std::min(person_roi.rows - fbox.y, fbox.height + 2 * h_expand);
                        if (fbox.width > 0 && fbox.height > 0) {
                            Mat face_aligned = person_roi(fbox).clone();
                            bool frontal = isFrontalFace(face_result[0].landmarks);
                            if (frontal) {
                                float ideal_area = CAMERA_WIDTH * CAMERA_HEIGHT * 0.15f;
                                float area_score = 1.0f / (1.0f + abs(current_area_4k - ideal_area) / ideal_area);
                                double current_score = current_clarity * 0.5 + area_score * 1000 * 0.5;
                                Track::FrameData frame_data;
                                frame_data.score = current_score;
                                frame_data.person_roi = person_roi.clone();
                                frame_data.face_roi = face_aligned.clone();
                                frame_data.has_face = true;
                                frame_data.clarity = current_clarity;
                                frame_data.area_ratio = area_ratio;
                                add_frame_candidate(t.id, frame_data);
                            }
                        }
                    }
                }
            }
        }
    }
}
