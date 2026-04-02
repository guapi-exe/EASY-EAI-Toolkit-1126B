#ifndef PERSON_SORT_H
#define PERSON_SORT_H

#include <vector>
#include <cstdint>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include "tinyekf.h"
#include <algorithm>
#include <cstdio>
#include <functional>
#include <unordered_set>

using namespace cv;

struct Detection {
    float x1, y1, x2, y2;
    cv::Mat roi;
    float prop;
};

struct Track {
    int id;
    ekf_t ekf;
    cv::Mat hist;
    cv::Rect2f bbox;
    cv::Rect2f smoothed_bbox;
    cv::Rect2f last_det_bbox;
    float prop;
    int age;
    int missed;
    int hits;
    bool active;
    bool confirmed;
    std::vector<float> bbox_history;
    float bbox_jitter;
    bool is_approaching;
    float best_area;
    double best_clarity;
    bool has_captured;

    struct FrameData {
        double score;
        cv::Mat person_roi;
        cv::Mat face_roi;
        bool has_face;
        bool is_frontal;
        uint8_t face_pose_level;
        bool strong_candidate;
        float yaw_abs;
        double clarity;
        float area_ratio;
        float person_occlusion;
        float face_edge_occlusion;
        float motion_ratio;
        float blur_severity;
    };
    std::vector<FrameData> frame_candidates;
};

void sort_init();
std::vector<Track> sort_update(const std::vector<Detection>& dets);
std::vector<Track> get_expiring_tracks();
void set_upload_callback(std::function<void(const cv::Mat&, int, const std::string&)> callback,
                         std::unordered_set<int>* person_ids,
                         std::unordered_set<int>* face_ids);
void set_max_frame_candidates(size_t maxFrameCandidates);
void add_frame_candidate(int track_id, const Track::FrameData& frame_data);

#endif
