#include "camera_task.h"
#include "main.h"
#include "person_detect.h"
#include "face_detect.h"
#include "sort_tracker.h"
extern "C" {
#include "log.h"
#include "camera.h"
}

using namespace cv;
using namespace std;

CameraTask::CameraTask(const string& personModel, const string& faceModel, int index)
    : personModelPath(personModel), faceModelPath(faceModel), cameraIndex(index), running(false) {}

CameraTask::~CameraTask() { stop(); }

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
    Mat gray, lap;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Laplacian(gray, lap, CV_64F);
    return mean(lap.mul(lap))[0];
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
    /*
    if (mipicamera_init(cameraIndex, CAMERA_WIDTH, CAMERA_HEIGHT, 0) != 0) {
        log_debug("CameraTask: Camera init failed");
        person_detect_release(personCtx);
        face_detect_release(faceCtx);
        return;
    }
    */
    if (usbcamera_init(cameraIndex, CAMERA_WIDTH, CAMERA_HEIGHT, 0) != 0) {
        log_debug("CameraTask: Camera init failed");
        person_detect_release(personCtx);
        face_detect_release(faceCtx);
        return;
    }

    vector<unsigned char> buffer(IMAGE_SIZE);
    while (running) {
        if (usbcamera_getframe(cameraIndex, reinterpret_cast<char*>(buffer.data())) != 0) continue;
        Mat frame(CAMERA_HEIGHT, CAMERA_WIDTH, CV_8UC3, buffer.data());
        if (frame.empty()) continue;
        processFrame(frame, personCtx, faceCtx);
    }

    mipicamera_exit(cameraIndex);
    person_detect_release(personCtx);
    face_detect_release(faceCtx);
}

void CameraTask::captureSnapshot() {
    std::vector<unsigned char> buffer(IMAGE_SIZE);
    if (usbcamera_getframe(cameraIndex, reinterpret_cast<char*>(buffer.data())) == 0) {
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
    detect_result_group_t detect_result_group;
    person_detect_run(personCtx, frame, &detect_result_group);

    vector<Detection> dets;
    for (int i=0; i<detect_result_group.count; i++) {
        detect_result_t& d = detect_result_group.results[i];
        if (d.prop < 0.7) continue;
        Rect roi(max(0,d.box.left), max(0,d.box.top),
                 min(CAMERA_WIDTH-1,d.box.right)-max(0,d.box.left),
                 min(CAMERA_HEIGHT-1,d.box.bottom)-max(0,d.box.top));
        if (roi.width <=0 || roi.height <=0) continue;
        Detection det; 
        det.roi = frame(roi).clone(); 
        det.x1 = roi.x; det.y1 = roi.y; 
        det.x2 = roi.x + roi.width; det.y2 = roi.y + roi.height; 
        det.prop = d.prop;
        dets.push_back(det);
    }

    vector<Track> tracks = sort_update(dets);

    for (auto& t : tracks) {
        Rect bbox((int)t.bbox.x, (int)t.bbox.y, (int)t.bbox.width, (int)t.bbox.height);
        if (bbox.width <=0 || bbox.height <=0) continue;
        Mat person_roi = frame(bbox).clone();

        vector<det> face_result;
        face_detect_run(faceCtx, person_roi, face_result);

        if (t.bbox_history.size() >= 5) {
            float area_now = t.bbox_history.back();
            float area_prev = t.bbox_history[t.bbox_history.size() - 5];
            float ratio = (area_now - area_prev) / (area_prev + 1e-6f);
            
            // 判断是否正在接近摄像机
            if (ratio > 0.1f) {
                t.is_approaching = true;
                //log_info("Track %d 正在接近摄像机 (面积变化: %.2f%%)", t.id, ratio*100);
            } else if (ratio < -0.1f) {
                t.is_approaching = false;
                //log_info("Track %d 正在远离摄像机 (面积变化: %.2f%%)", t.id, ratio*100);
            }
        }

        if (t.is_approaching && !t.has_captured && !face_result.empty()) {
            double current_clarity = computeFocusMeasure(person_roi);
            float current_area = t.bbox.width * t.bbox.height;

            float ideal_area = CAMERA_WIDTH * CAMERA_HEIGHT * 0.15f; // 期望面积约占画面15%
            float area_score = 1.0f / (1.0f + abs(current_area - ideal_area) / ideal_area);
            
            double current_score = current_clarity * 0.5 + area_score * 1000 * 0.5;

            // 计算人员在画面中的占比
            float area_ratio = current_area / (CAMERA_WIDTH * CAMERA_HEIGHT);

            // 当人员占比大于5%且有人脸时，记录候选帧
            if (area_ratio > 0.05f && current_clarity > 100) {
                // 处理人脸框
                Rect fbox = cv::Rect(face_result[0].box);
                int w_expand = static_cast<int>(fbox.width * 0.5 / 2.0);
                int h_expand = static_cast<int>(fbox.height * 0.5 / 2.0);
                fbox.x = std::max(0, fbox.x - w_expand);
                fbox.y = std::max(0, fbox.y - h_expand);
                fbox.width = std::min(person_roi.cols - fbox.x, fbox.width + 2 * w_expand);
                fbox.height = std::min(person_roi.rows - fbox.y, fbox.height + 2 * h_expand);

                if (fbox.width > 0 && fbox.height > 0) {
                    Mat face_aligned = person_roi(fbox).clone();
                    if (computeFocusMeasure(face_aligned) > 100) {
                        // 记录候选帧数据
                        Track::FrameData frame_data;
                        frame_data.score = current_score;
                        frame_data.person_roi = person_roi.clone();
                        frame_data.face_roi = face_aligned.clone();
                        frame_data.has_face = true;
                        frame_data.clarity = current_clarity;
                        frame_data.area_ratio = area_ratio;
                        
                        if (t.frame_candidates.size() < 20) {
                            t.frame_candidates.push_back(frame_data);
                        } else {
                            auto min_it = std::min_element(t.frame_candidates.begin(), t.frame_candidates.end(),
                                [](const Track::FrameData& a, const Track::FrameData& b) {
                                    return a.score < b.score;
                                });
                            
                            if (current_score > min_it->score) {
                                *min_it = frame_data;  
                                //log_debug("Track %d 替换低分帧 (新分数: %.2f > 旧分数: %.2f)", t.id, current_score, min_it->score);
                            } else {
                                //log_debug("Track %d 跳过低分帧 (分数: %.2f 不高于最低分: %.2f)", t.id, current_score, min_it->score);
                                continue; 
                            }
                        }
                        
                        log_debug("Track %d 记录候选帧 (清晰度: %.2f, 面积占比: %.2f%%, 综合评分: %.2f)", 
                                 t.id, current_clarity, area_ratio*100, current_score);
                    }
                }
            }
        }
    }
}
