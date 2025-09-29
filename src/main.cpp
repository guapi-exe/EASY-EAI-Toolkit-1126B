#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <sys/time.h>
#include <string>
#include "person_detect.h"
#include "face_detect.h"
#include "person_data.h"
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
    log_debug("person_detect_init & face_detect_init done.\n");

    ret = mipicamera_init(cameraIndex, CAMERA_WIDTH, CAMERA_HEIGHT, 0);
    if (ret) { log_debug("mipicamera_init failed\n"); return -1; }

    pbuf = (char *)malloc(IMAGE_SIZE);
    if (!pbuf) { log_debug("malloc failed\n"); ret = -1; goto exit_cam; }

    while (true) {
        frame_id++;
        ret = mipicamera_getframe(cameraIndex, pbuf);
        if (ret) { log_debug("getframe failed\n"); break; }

        Mat frame(CAMERA_HEIGHT, CAMERA_WIDTH, CV_8UC3, pbuf);
        if (frame.empty()) { log_debug("frame empty\n"); break; }

        detect_result_group_t detect_result_group;
        struct timeval start, end;
        gettimeofday(&start, NULL);

        person_detect_run(person_ctx, frame, &detect_result_group);

        gettimeofday(&end, NULL);
        float time_use = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
        log_debug("Person Detection time: %f ms, count=%d\n", time_use / 1000, detect_result_group.count);

        std::vector<Detection> dets;
        for (int i=0;i<detect_result_group.count;i++){
            detect_result_t& d = detect_result_group.results[i];
            if (d.prop < 0.4) continue;
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

            // **新人员出现时截图并抓取人脸**
            if (t.age == 1) {
                log_debug("New person appeared: ID=%d\n", t.id);

                Mat person_roi = frame(Rect(x1, y1, w, h)).clone();
                std::vector<det> face_result;
                face_detect_run(face_ctx, person_roi, face_result);

                if (!face_result.empty()) {
                    int fx = std::max(0, (int)face_result[0].box.x);
                    int fy = std::max(0, (int)face_result[0].box.y);
                    int fw = std::min((int)face_result[0].box.width, person_roi.cols - fx);
                    int fh = std::min((int)face_result[0].box.height, person_roi.rows - fy);

                    if (fw > 0 && fh > 0) {
                        Mat face_aligned = person_roi(Rect(fx, fy, fw, fh)).clone();

                        // 模糊检测
                        double fm = compute_focus_measure(face_aligned);
                        if (fm > 100) { // 阈值可调
                            char face_name[128];
                            snprintf(face_name, sizeof(face_name), "person_%05d_face_%d.jpg", frame_id, t.id);
                            imwrite(face_name, face_aligned);
                            log_debug("Saved aligned face: %s\n", face_name);
                        } else {
                            log_debug("Face too blurry, skip save. FM=%f\n", fm);
                        }
                    }
                } else {
                    // 无人脸，则保存整个人体ROI
                    double fm = compute_focus_measure(person_roi);
                    if (fm > 100) {
                        char person_name[128];
                        snprintf(person_name, sizeof(person_name), "person_%05d_id_%d.jpg", frame_id, t.id);
                        imwrite(person_name, person_roi);
                        log_debug("Saved person ROI (no face): %s\n", person_name);
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
        log_debug("Usage: %s <person_model> <face_model>\n", argv[0]);
        return -1;
    }

    const char *person_model_path = argv[1];
    const char *face_model_path   = argv[2];

    sort_init();
    return run_person_detect_video(person_model_path, face_model_path, 22);
}
