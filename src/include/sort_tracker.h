#ifndef PERSON_SORT_H
#define PERSON_SORT_H

#include <vector>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include "tinyekf.h"
#include <algorithm>
#include <cstdio>


using namespace cv;

struct Detection {
    float x1, y1, x2, y2;
    cv::Mat roi;  // 用于计算颜色直方图
    float prop;
};

struct Track {
    int id;
    ekf_t ekf;    // 使用 tinyEKF
    cv::Mat hist; // 颜色直方图
    cv::Rect2f bbox;
    float prop;
    int age;
    int missed; // 丢失帧数
    int hits;   // 命中次数，用于稳定性评估  
    bool active;
    bool confirmed; // 是否已确认的track
};

void sort_init();
std::vector<Track> sort_update(const std::vector<Detection>& dets);

#endif
