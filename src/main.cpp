#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <sys/time.h>
#include <string>
#include "person_detect.h"
#include "face_detect.h"
#include "person_data.h"
#include "sort_tracker.h"
extern "C" {
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

/* 人形识别视频流处理 */
int run_person_detect_video(const char *model_path, int cameraIndex = 0)
{
    rknn_context ctx;
    int ret = 0;
    char *pbuf = nullptr;
    int frame_id = 0;

    person_detect_init(&ctx, model_path);
    printf("person_detect_init done.\n");

    ret = mipicamera_init(cameraIndex, CAMERA_WIDTH, CAMERA_HEIGHT, 0);
    if (ret) { printf("mipicamera_init failed\n"); return -1; }
    printf("mipicamera_init done.\n");

    pbuf = (char *)malloc(IMAGE_SIZE);
    if (!pbuf) { printf("malloc failed\n"); ret = -1; goto exit_cam; }

    while (true) {
        frame_id++;
        ret = mipicamera_getframe(cameraIndex, pbuf);
        if (ret) { printf("getframe failed\n"); break; }

        Mat frame(CAMERA_HEIGHT, CAMERA_WIDTH, CV_8UC3, pbuf);
        if (frame.empty()) { printf("frame empty\n"); break; }

        detect_result_group_t detect_result_group;
        struct timeval start, end;
        gettimeofday(&start, NULL);

        person_detect_run(ctx, frame, &detect_result_group);

        gettimeofday(&end, NULL);
        float time_use = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
        printf("Person Detection time: %f ms, count=%d\n", time_use / 1000, detect_result_group.count);

        std::vector<Detection> dets;
        for (int i = 0; i < detect_result_group.count; i++) {
            detect_result_t* det_result = &(detect_result_group.results[i]);
            if (det_result->prop < 0.4) continue;

            Detection d;
            d.x1 = det_result->box.left;
            d.y1 = det_result->box.top;
            d.x2 = det_result->box.right;
            d.y2 = det_result->box.bottom;
            d.score = det_result->prop;
            dets.push_back(d);
        }

        std::vector<Track> tracks = sort_update(dets);

        for (const auto &t : tracks) {
            int x1 = std::max(0, (int)std::round(t.x1));
            int y1 = std::max(0, (int)std::round(t.y1));
            int x2 = std::min(CAMERA_WIDTH - 1, (int)std::round(t.x2));
            int y2 = std::min(CAMERA_HEIGHT - 1, (int)std::round(t.y2));

            int w = x2 - x1;
            int h = y2 - y1;
            if (w <= 0 || h <= 0) continue;

            cv::rectangle(frame, cv::Rect(x1, y1, w, h), cv::Scalar(0, 255, 0), 2);

            char label[64];
            sprintf(label, "ID:%d", t.id);
            cv::putText(frame, label, cv::Point(x1, y1 - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);

            if (t.missed == 0 && (frame_id % 30 == 0)) {
                cv::Mat roi = frame(cv::Rect(x1, y1, w, h)).clone();  
                char person_img_name[128];
                snprintf(person_img_name, sizeof(person_img_name),
                        "track_person_%05d_%d.jpg", frame_id, t.id);
                cv::imwrite(person_img_name, roi);
                printf("Saved track person: %s\n", person_img_name);
            }
        }
                
        if (detect_result_group.count && frame_id % 30 == 0) {
            /*
            char save_name[128];
            snprintf(save_name, sizeof(save_name), "person_frame_%05d.jpg", frame_id);
            imwrite(save_name, frame);

            for (int i = 0; i < (int)result.size(); i++) 
            { 
                int x = std::max(0, (int)result[i].box.x); 
                int y = std::max(0, (int)result[i].box.y); 
                int w = std::min((int)result[i].box.width, CAMERA_WIDTH - x); 
                int h = std::min((int)result[i].box.height, CAMERA_HEIGHT - y); 
                Mat person_roi = frame(Rect(x, y, w, h)); char person_img_name[128]; 
                snprintf(person_img_name, sizeof(person_img_name), "person_%05d_%d.jpg", frame_id, i); 
                imwrite(person_img_name, person_roi); printf("Saved person crop: %s\n", person_img_name); 
            }
            */
            
        }
    }

    if (pbuf) free(pbuf);
exit_cam:
    mipicamera_exit(cameraIndex);
    person_detect_release(ctx);
    return ret;
}

/* 人脸识别视频流处理 */
int run_face_detect_video(const char *model_path, int cameraIndex = 0)
{
    rknn_context ctx;
    int ret = 0;
    char *pbuf = nullptr;
    int frame_id = 0;

    face_detect_init(&ctx, model_path);
    printf("face_detect_init done.\n");

    ret = mipicamera_init(cameraIndex, CAMERA_WIDTH, CAMERA_HEIGHT, 0);
    if (ret) { printf("mipicamera_init failed\n"); return -1; }
    printf("mipicamera_init done.\n");

    pbuf = (char *)malloc(IMAGE_SIZE);
    if (!pbuf) { printf("malloc failed\n"); ret = -1; goto exit_cam; }

    while (true) {
        ret = mipicamera_getframe(cameraIndex, pbuf);
        if (ret) { printf("getframe failed\n"); break; }

        Mat frame(CAMERA_HEIGHT, CAMERA_WIDTH, CV_8UC3, pbuf);
        if (frame.empty()) { printf("frame empty\n"); break; }

        std::vector<det> result;
        struct timeval start, end;
        gettimeofday(&start, NULL);
        face_detect_run(ctx, frame, result);
        gettimeofday(&end, NULL);
        float time_use = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
        printf("Face Detection time: %f ms, Faces=%d\n", time_use/1000, (int)result.size());

        for (int i = 0; i < (int)result.size(); i++) {
            int x = (int)(result[i].box.x);
            int y = (int)(result[i].box.y);
            int w = (int)(result[i].box.width);
            int h = (int)(result[i].box.height);
            rectangle(frame, Rect(x, y, w, h), Scalar(0, 255, 0), 2);

            for (int j = 0; j < (int)result[i].landmarks.size(); ++j) {
                circle(frame, Point((int)result[i].landmarks[j].x, (int)result[i].landmarks[j].y), 2, Scalar(225, 0, 225), 2);
            }
        }

        frame_id++;
        if (result.size() && frame_id % 30 == 0) {
            
            /*
            char save_name[128];
            snprintf(save_name, sizeof(save_name), "face_frame_%05d.jpg", frame_id);
            imwrite(save_name, frame);
            for (int i = 0; i < (int)result.size(); i++) 
            { 
                int x = std::max(0, (int)result[i].box.x); 
                int y = std::max(0, (int)result[i].box.y); 
                int w = std::min((int)result[i].box.width, CAMERA_WIDTH - x); 
                int h = std::min((int)result[i].box.height, CAMERA_HEIGHT - y); 
                Mat face_roi = frame(Rect(x, y, w, h)); char face_img_name[128]; 
                snprintf(face_img_name, sizeof(face_img_name), "face_%05d_%d.jpg", frame_id, i); 
                imwrite(face_img_name, face_roi); printf("Saved face crop: %s\n", face_img_name); 
            }
            */
            
        }
    }

    if (pbuf) free(pbuf);
exit_cam:
    mipicamera_exit(cameraIndex);
    face_detect_release(ctx);
    return ret;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("Usage: %s <mode> [model_path]\n", argv[0]);
        printf("mode: person | face\n");
        return -1;
    }

    string mode = argv[1];
    const char *model_path = (argc >= 3) ? argv[2] : nullptr;
    sort_init();
    if (mode == "person") {
        if (!model_path) model_path = "person_detect.model";
        return run_person_detect_video(model_path, 22);
    } else if (mode == "face") {
        if (!model_path) model_path = "face_detect.model";
        return run_face_detect_video(model_path, 22);
    } else {
        printf("Unknown mode: %s\n", mode.c_str());
        return -1;
    }
}
