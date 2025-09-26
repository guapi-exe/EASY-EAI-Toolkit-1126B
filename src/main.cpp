#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <sys/time.h>
#include "face_detect.h"
extern "C" {
#include "camera.h"
}

using namespace cv;
using namespace std;

#define CAMERA_WIDTH    1920
#define CAMERA_HEIGHT   1080
#define IMGRATIO        3
#define IMAGE_SIZE      (CAMERA_WIDTH*CAMERA_HEIGHT*IMGRATIO)

int main(int argc, char **argv)
{
    rknn_context ctx;
    int ret;
    int cameraIndex = 0; // 假设摄像头索引为0
    char *pbuf = NULL;

    face_detect_init(&ctx, "face_detect.model");
    printf("face_detect_init done.\n");

    ret = mipicamera_init(cameraIndex, CAMERA_WIDTH, CAMERA_HEIGHT, 0);
    if (ret) {
        printf("error: mipicamera_init failed\n");
        goto exit_rknn;
    }
    printf("mipicamera_init done.\n");

    pbuf = (char *)malloc(IMAGE_SIZE);
    if (!pbuf) {
        printf("error: malloc failed\n");
        ret = -1;
        goto exit_cam;
    }
    while (true) {
        ret = mipicamera_getframe(cameraIndex, pbuf);
        if (ret) {
            printf("error: mipicamera_getframe failed\n");
            break;
        }

        Mat frame(CAMERA_HEIGHT, CAMERA_WIDTH, CV_8UC3, pbuf);
        if (frame.empty()) {
            printf("error: create cv::Mat from buffer failed\n");
            break;
        }

        std::vector<det> result;
        struct timeval start, end;
        gettimeofday(&start, NULL);

        face_detect_run(ctx, frame, result);

        gettimeofday(&end, NULL);
        float time_use = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
        printf("Detection time: %f ms\n", time_use / 1000);
        printf("Faces found: %d\n", (int)result.size());

        // d. 在帧上绘制检测结果
        for (int i = 0; i < (int)result.size(); i++) {
            int x = (int)(result[i].box.x);
            int y = (int)(result[i].box.y);
            int w = (int)(result[i].box.width);
            int h = (int)(result[i].box.height);
            rectangle(frame, Rect(x, y, w, h), Scalar(0, 255, 0), 2, 8, 0);

            for (int j = 0; j < (int)result[i].landmarks.size(); ++j) {
                cv::circle(frame, cv::Point((int)result[i].landmarks[j].x, (int)result[i].landmarks[j].y), 2, cv::Scalar(225, 0, 225), 2, 8);
            }
        }

        //cv::imshow("Face Detection", frame);

    }

    if (pbuf) {
        free(pbuf);
    }
    cv::destroyAllWindows();

exit_cam:
    mipicamera_exit(cameraIndex); 
    printf("mipicamera_exit done.\n");
exit_rknn:
    face_detect_release(ctx);
    printf("face_detect_release done.\n");

    return ret;
}
