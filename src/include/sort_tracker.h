#ifndef PERSON_SORT_H
#define PERSON_SORT_H

#include <vector>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include "tinyekf.h"
#include <algorithm>
#include <cstdio>

struct Detection {
    float x1, y1, x2, y2;
    cv::Mat roi;  // 用于计算颜色直方图
};

struct Track {
    int id;
    ekf_t ekf;    // 使用 tinyEKF
    cv::Mat hist; // 颜色直方图
    cv::Rect2f bbox;
    int age;
    int missed; // 丢失帧数
    bool active;
};

void sort_init();
std::vector<Track> sort_update(const std::vector<Detection>& dets);

#endif
