#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <sys/time.h>
#include <string>
#include "person_detect.h"
#include "face_detect.h"
#include "person_data.h"
#include "send_data.h"
#include "sort_tracker.h"
#include <unordered_set>
extern "C" {
#include "log.h"
#include "camera.h"
}

using namespace cv;
using namespace std;
std::unordered_set<int> captured_ids;
std::unordered_set<int> captured_person_ids;

#define CAMERA_WIDTH    1920
#define CAMERA_HEIGHT   1080
#define IMGRATIO        3
#define IMAGE_SIZE      (CAMERA_WIDTH*CAMERA_HEIGHT*IMGRATIO)

/* 用于绘制人形识别框 */
static Scalar colorArray[10] = {
    Scalar(255, 0, 0, 255), Scalar(0, 255, 0, 255), Scalar(0,0,139,255),
    Scalar(0,100,0,255), Scalar(139,139,0,255), Scalar(209,206,0,255),
    Scalar(0,127,255,255), Scalar(139,61,72,255), Scalar(0,255,0,255),
    Scalar(255,0,0,255),
};

int plot_one_box(Mat src, int x1, int x2, int y1, int y2, const char *label, char colour)
{
    int tl = round(0.002 * (src.rows + src.cols) / 2) + 1;
    rectangle(src, cv::Point(x1, y1), cv::Point(x2, y2), colorArray[(unsigned char)colour], 3);
    int tf = max(tl - 1, 1);

    int base_line = 0;
    cv::Size t_size = getTextSize(label, FONT_HERSHEY_SIMPLEX, (float)tl/3, tf, &base_line);
    int x3 = x1 + t_size.width;
    int y3 = y1 - t_size.height - 3;
    rectangle(src, cv::Point(x1, y1), cv::Point(x3, y3), colorArray[(unsigned char)colour], -1);
    putText(src, label, cv::Point(x1, y1 - 2), FONT_HERSHEY_SIMPLEX, (float)tl/3, cv::Scalar(255, 255, 255, 255), tf, 8);
    return 0;
}

// 计算图像清晰度
double compute_focus_measure(const Mat& img) {
    Mat gray, lap;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Laplacian(gray, lap, CV_64F);
    return mean(lap.mul(lap))[0];
}

/* 人形识别视频流处理 */
int run_person_detect_video(const char *person_model_path, const char *face_model_path, int cameraIndex = 0)
{
    rknn_context person_ctx, face_ctx;
    int ret = 0;
    char *pbuf = nullptr;
    int frame_id = 0;

    person_detect_init(&person_ctx, person_model_path);
    face_detect_init(&face_ctx, face_model_path);
    log_debug("person_detect_init & face_detect_init done.");

    ret = mipicamera_init(cameraIndex, CAMERA_WIDTH, CAMERA_HEIGHT, 0);
    if (ret) { log_debug("mipicamera_init failed"); return -1; }

    pbuf = (char *)malloc(IMAGE_SIZE);
    if (!pbuf) { log_debug("malloc failed"); ret = -1; goto exit_cam; }

    while (true) {
        frame_id++;
        ret = mipicamera_getframe(cameraIndex, pbuf);
        if (ret) { log_debug("getframe failed"); break; }

        Mat frame(CAMERA_HEIGHT, CAMERA_WIDTH, CV_8UC3, pbuf);
        if (frame.empty()) { log_debug("frame empty"); break; }

        detect_result_group_t detect_result_group;
        struct timeval start, end;
        gettimeofday(&start, NULL);

        person_detect_run(person_ctx, frame, &detect_result_group);

        gettimeofday(&end, NULL);
        float time_use = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
        log_debug("Person Detection time: %f ms, count=%d", time_use / 1000, detect_result_group.count);

        std::vector<Detection> dets;
        for (int i=0;i<detect_result_group.count;i++){
            detect_result_t& d = detect_result_group.results[i];
            if (d.prop < 0.6) continue;
            int x1 = std::max(0, d.box.left);
            int y1 = std::max(0, d.box.top);
            int x2 = std::min(CAMERA_WIDTH - 1, d.box.right);
            int y2 = std::min(CAMERA_HEIGHT - 1, d.box.bottom);

            int w = x2 - x1;
            int h = y2 - y1;
            if (w <= 0 || h <= 0) continue; // 无效框直接跳过

            cv::Rect roi_rect(x1, y1, w, h);
            Detection det;
            det.roi = frame(roi_rect).clone();
            det.x1 = x1;
            det.y1 = y1;
            det.x2 = x2;
            det.y2 = y2;
            det.prop = d.prop;

            dets.push_back(det);
        }

        std::vector<Track> tracks = sort_update(dets);

        for (auto &t : tracks) {
            int x1 = std::max(0, (int)std::round(t.bbox.x));
            int y1 = std::max(0, (int)std::round(t.bbox.y));
            int x2 = std::min(CAMERA_WIDTH - 1, (int)std::round(t.bbox.x + t.bbox.width));
            int y2 = std::min(CAMERA_HEIGHT - 1, (int)std::round(t.bbox.y + t.bbox.height));
            int w = x2 - x1;
            int h = y2 - y1;
            if (w <= 0 || h <= 0) continue;

            rectangle(frame, cv::Rect(x1, y1, w, h), cv::Scalar(0, 255, 0), 2);
            char label[64]; sprintf(label,"ID:%d", t.id);
            putText(frame, label, Point(x1, y1-5), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255,255,255), 2);
            Mat person_roi;
            if(captured_person_ids.find(t.id) == captured_person_ids.end() || captured_ids.find(t.id) == captured_ids.end()){
                person_roi = frame(Rect(x1, y1, w, h)).clone();
            }

            // **持续对未抓取到的人员进行抓拍**
            if (captured_person_ids.find(t.id) == captured_person_ids.end()) {
                double confidence = t.prop;
                if (confidence > 0.8) { // 置信度判断
                    double fm = compute_focus_measure(person_roi);
                    if (fm > 100) {
                        if (captured_person_ids.find(t.id) == captured_person_ids.end()) {
                            std::string json = build_json(person_roi, t.id, "person");
                            log_debug("Captured person JSON: %s", json.c_str());
                            captured_person_ids.insert(t.id);
                            //snprintf(person_name, sizeof(person_name), "person_%05d_id_%d.jpg", frame_id, t.id);
                            //imwrite(person_name, person_roi);
                            //log_debug("Saved person ROI (no face): %s", person_name);
                        }
                    } else {
                        log_debug("Person too blurry, skip save. FM=%f", fm);
                    }
                }
            }
            if (captured_ids.find(t.id) == captured_ids.end()) {
                log_debug("Trying face capture for ID=%d (age=%d)", t.id, t.age);

                std::vector<det> face_result;
                face_detect_run(face_ctx, person_roi, face_result);

                bool face_captured = false;

                if (!face_result.empty()) {
                    int fx = std::max(0, (int)face_result[0].box.x);
                    int fy = std::max(0, (int)face_result[0].box.y);
                    int fw = std::min((int)face_result[0].box.width, person_roi.cols - fx);
                    int fh = std::min((int)face_result[0].box.height, person_roi.rows - fy);

                    if (fw > 0 && fh > 0) {
                        Mat face_aligned = person_roi(Rect(fx, fy, fw, fh)).clone();

                        double fm = compute_focus_measure(face_aligned);
                        if (fm > 100) {
                            char face_name[128];
                            std::string json = build_json(face_aligned, t.id, "face");
                            log_debug("Captured face JSON: %s", json.c_str());
                            //snprintf(face_name, sizeof(face_name), "person_%05d_face_%d.jpg", frame_id, t.id);
                            //imwrite(face_name, face_aligned);
                            //log_debug("Saved aligned face: %s", face_name);

                            captured_ids.insert(t.id);
                            face_captured = true;
                        } else {
                            log_debug("Face too blurry, skip save. FM=%f", fm);
                        }
                    }
                }
            }
        }
    }

    if (pbuf) free(pbuf);
exit_cam:
    mipicamera_exit(cameraIndex);
    person_detect_release(person_ctx);
    face_detect_release(face_ctx);
    return ret;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        log_debug("Usage: %s <person_model> <face_model>", argv[0]);
        return -1;
    }

    const char *person_model_path = argv[1];
    const char *face_model_path   = argv[2];

    sort_init();
    return run_person_detect_video(person_model_path, face_model_path, 22);
}
