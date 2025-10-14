#include "camera_task.h"
#include "person_detect.h"
#include "face_detect.h"
#include "sort_tracker.h"
extern "C" {
#include "log.h"
#include "camera.h"
}

using namespace cv;
using namespace std;

#define CAMERA_WIDTH    1920
#define CAMERA_HEIGHT   1080
#define IMGRATIO        3
#define IMAGE_SIZE      (CAMERA_WIDTH*CAMERA_HEIGHT*IMGRATIO)

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
        if (d.prop < 0.6) continue;
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

        if (!face_result.empty() && computeFocusMeasure(person_roi) > 100) {
            // 原始人脸框
            Rect fbox = cv::Rect(face_result[0].box);

            int w_expand = static_cast<int>(fbox.width * 0.5 / 2.0); // 左右各扩大一半
            int h_expand = static_cast<int>(fbox.height * 0.5 / 2.0); // 上下各扩大一半
            fbox.x = std::max(0, fbox.x - w_expand);
            fbox.y = std::max(0, fbox.y - h_expand);
            fbox.width = std::min(person_roi.cols - fbox.x, fbox.width + 2 * w_expand);
            fbox.height = std::min(person_roi.rows - fbox.y, fbox.height + 2 * h_expand);

            if (fbox.width > 0 && fbox.height > 0) {
                Mat face_aligned = person_roi(fbox).clone();
                if (computeFocusMeasure(face_aligned) > 100) {
                    if (capturedPersonIds.find(t.id) == capturedPersonIds.end() &&
                        capturedFaceIds.find(t.id) == capturedFaceIds.end()) {
                        if (uploadCallback) {
                            uploadCallback(frame.clone(), 0, "all");//test
                            uploadCallback(person_roi, t.id, "person");
                            uploadCallback(face_aligned, t.id, "face");
                        }
                        capturedPersonIds.insert(t.id);
                        capturedFaceIds.insert(t.id);
                    }
                }
            }
        }
    }
}
