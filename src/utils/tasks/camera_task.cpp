#include "camera_task.h"
#include "person_detect.h"
#include "face_detect.h"
#include "sort_tracker.h"
#include "log.h"
#include "camera.h"

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

double CameraTask::computeFocusMeasure(const Mat& img) {
    Mat gray, lap;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Laplacian(gray, lap, CV_64F);
    return mean(lap.mul(lap))[0];
}

void CameraTask::run() {
    rknn_context personCtx, faceCtx;
    person_detect_init(&personCtx, personModelPath.c_str());
    face_detect_init(&faceCtx, faceModelPath.c_str());
    sort_init();

    if (mipicamera_init(cameraIndex, CAMERA_WIDTH, CAMERA_HEIGHT, 0) != 0) {
        log_debug("CameraTask: Camera init failed");
        return;
    }

    std::vector<unsigned char> buffer(IMAGE_SIZE);
    while (running) {
        if (mipicamera_getframe(cameraIndex, reinterpret_cast<char*>(buffer.data())) != 0) continue;

        cv::Mat frame(CAMERA_HEIGHT, CAMERA_WIDTH, CV_8UC3, buffer.data());
        if (frame.empty()) continue;

        processFrame(frame);  
    }

    mipicamera_exit(cameraIndex);
    person_detect_release(personCtx);
    face_detect_release(faceCtx);
}


void CameraTask::processFrame(const Mat& frame) {
    detect_result_group_t detect_result_group;
    person_detect_run(0, frame, &detect_result_group);

    vector<Detection> dets;
    for (int i=0; i<detect_result_group.count; i++) {
        detect_result_t& d = detect_result_group.results[i];
        if (d.prop < 0.6) continue;
        Rect roi(max(0,d.box.left), max(0,d.box.top),
                 min(CAMERA_WIDTH-1,d.box.right)-max(0,d.box.left),
                 min(CAMERA_HEIGHT-1,d.box.bottom)-max(0,d.box.top));
        if (roi.width <=0 || roi.height <=0) continue;
        Detection det; det.roi = frame(roi).clone(); det.x1=roi.x; det.y1=roi.y; det.x2=roi.x+roi.width; det.y2=roi.y+roi.height; det.prop=d.prop;
        dets.push_back(det);
    }

    vector<Track> tracks = sort_update(dets);
    for (auto& t : tracks) {
        Rect bbox((int)t.bbox.x, (int)t.bbox.y, (int)t.bbox.width, (int)t.bbox.height);
        if (bbox.width<=0 || bbox.height<=0) continue;
        Mat person_roi = frame(bbox).clone();

        // 捕获人形
        if (capturedPersonIds.find(t.id) == capturedPersonIds.end() && computeFocusMeasure(person_roi) > 100) {
            if (uploadCallback) uploadCallback(person_roi, t.id, "person");
            capturedPersonIds.insert(t.id);
        }

        // 捕获人脸
        if (capturedFaceIds.find(t.id) == capturedFaceIds.end()) {
            vector<det> face_result;
            face_detect_run(0, person_roi, face_result);
            if (!face_result.empty()) {
                cv::Rect fbox = cv::Rect(face_result[0].box) & cv::Rect(0, 0, person_roi.cols, person_roi.rows);
                if (fbox.width>0 && fbox.height>0) {
                    Mat face_aligned = person_roi(fbox).clone();
                    if (computeFocusMeasure(face_aligned)>100) {
                        if (uploadCallback) uploadCallback(face_aligned, t.id, "face");
                        capturedFaceIds.insert(t.id);
                    }
                }
            }
        }
    }
}
