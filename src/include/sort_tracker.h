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
    bool allow_new_track{true};
};

enum class TrackTrajectoryDirection : uint8_t {
    Unknown = 0,
    Approaching = 1,
    Leaving = 2,
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
    std::vector<cv::Point2f> trajectory_history;
    std::vector<float> trajectory_area_history;
    float bbox_jitter;
    bool is_approaching;
    TrackTrajectoryDirection trajectory_direction;
    float trajectory_score;
    float max_area_ratio;
    bool has_reversed;
    float peak_bottom;
    float peak_area;
    double best_clarity;
    bool has_captured;

    struct FrameData {
        double score;
        cv::Mat person_roi;
        cv::Mat face_roi;
        bool has_face;
        cv::Rect face_bbox_720p;
        float face_confidence{0.0f};
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

struct TrackSnapshot {
    int id{0};
    cv::Rect2f bbox;
    cv::Rect2f smoothed_bbox;
    float prop{0.0f};
    int missed{0};
    int hits{0};
    bool confirmed{false};
    float bbox_jitter{0.0f};
    bool is_approaching{false};
    TrackTrajectoryDirection trajectory_direction{TrackTrajectoryDirection::Unknown};
    float trajectory_score{0.0f};
    float max_area_ratio{0.0f};
    bool has_reversed{false};
    float peak_bottom{0.0f};
    float peak_area{0.0f};
    bool has_captured{false};
    bool has_face{false};
    cv::Rect face_bbox_720p;
    float face_confidence{0.0f};
    std::vector<cv::Point2f> trajectory_history;
};

void sort_init();
std::vector<TrackSnapshot> sort_update(const std::vector<Detection>& dets);
std::vector<TrackSnapshot> sort_predict_only();
std::vector<TrackSnapshot> get_expiring_tracks();
void set_upload_callback(std::function<void(const cv::Mat&, int, const std::string&)> callback,
                         std::unordered_set<int>* person_ids,
                         std::unordered_set<int>* face_ids);
void set_max_frame_candidates(size_t maxFrameCandidates);
void set_capture_sort_preferences(float minAreaRatio,
                                  float nearAreaRatio,
                                  float maxPersonOcclusion,
                                  bool requireApproach);
void add_frame_candidate(int track_id, const Track::FrameData& frame_data);

#endif
