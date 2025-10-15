#ifndef PERSON_SORT_H
#define PERSON_SORT_H

#include <vector>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include "tinyekf.h"
#include <algorithm>
#include <cstdio>
#include <functional>
#include <set>


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
    std::vector<float> bbox_history; // 检测框面积历史
    bool is_approaching; // 是否正在接近摄像机
    float best_area; // 最佳拍照时的面积
    double best_clarity; // 最佳拍照时的清晰度
    bool has_captured; // 是否已经拍照
    
    // 存储每帧的评分和图像数据
    struct FrameData {
        double score;
        cv::Mat person_roi;
        cv::Mat face_roi;
        bool has_face;
        double clarity;
        float area_ratio;
    };
    std::vector<FrameData> frame_candidates; // 候选帧数据
};

void sort_init();
std::vector<Track> sort_update(const std::vector<Detection>& dets);
std::vector<Track> get_expiring_tracks(); // 获取即将过期的tracks
void set_upload_callback(std::function<void(const cv::Mat&, int, const std::string&)> callback,
                        std::set<int>* person_ids, std::set<int>* face_ids);

#endif
