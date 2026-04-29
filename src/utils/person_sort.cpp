#include "sort_tracker.h"
#include <vector>
#include "main.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include "tinyekf.h"
#include <functional>
#include <mutex>
#include <unordered_set>
extern "C" {
#include "log.h"
}

static std::vector<Track> tracks;
static int next_id = 1;
static std::mutex tracks_mutex;
static size_t g_max_frame_candidates = CAPTURE_MAX_FRAME_CANDIDATES;

struct PendingTrack {
    cv::Rect2f bbox;
    cv::Mat hist;
    float prop;
    int hits;
    int ttl;
};

struct LostTrack {
    int id;
    cv::Rect2f bbox;
    cv::Mat hist;
    int ttl;
};

struct RecentCapture {
    int id;
    cv::Rect2f bbox;
    cv::Mat hist;
    int ttl;
};

static std::vector<LostTrack> lost_tracks;
static std::vector<RecentCapture> recent_captures;
static std::vector<PendingTrack> pending_tracks;
static constexpr int LOST_TRACK_TTL = 35;
static constexpr int RECENT_CAPTURE_TTL = 75;
static constexpr int PENDING_TRACK_TTL = 4;
static constexpr int PENDING_TRACK_HITS_REQUIRED = 2;
static constexpr size_t BBOX_HISTORY_LIMIT = 40;
static constexpr float TRACK_SMOOTH_CENTER_ALPHA = 0.30f;
static constexpr float TRACK_SMOOTH_SIZE_ALPHA = 0.16f;
static constexpr float TRACK_RECOVER_CENTER_ALPHA = 0.48f;
static constexpr float TRACK_RECOVER_SIZE_ALPHA = 0.30f;
static constexpr float TRACK_NEW_CENTER_ALPHA = 0.60f;
static constexpr float TRACK_NEW_SIZE_ALPHA = 0.38f;
static constexpr float TRACK_BBOX_JITTER_ALPHA = 0.28f;
static float g_capture_min_area_ratio = CAPTURE_MIN_AREA_RATIO;
static float g_capture_near_area_ratio = CAPTURE_NEAR_AREA_RATIO;
static float g_capture_max_person_occlusion = CAPTURE_MAX_PERSON_OCCLUSION;
static bool g_capture_require_approach = CAPTURE_REQUIRE_APPROACH != 0;
static constexpr size_t TRAJECTORY_EDGE_WINDOW_MAX = 12;
static constexpr float TRAJECTORY_MIN_BOTTOM_TRAVEL = 0.015f;
static constexpr float TRAJECTORY_MIN_AREA_TRAVEL = 0.008f;
static constexpr float TRAJECTORY_SCORE_THRESHOLD = 0.08f;
static constexpr float TRAJECTORY_RETURN_SCORE_THRESHOLD = 0.06f;

struct TrackTrajectoryDecision {
    TrackTrajectoryDirection direction{TrackTrajectoryDirection::Unknown};
    float score{0.0f};
    float bottom_start{0.0f};
    float bottom_end{0.0f};
    float area_start{0.0f};
    float area_end{0.0f};
    float bottom_travel{0.0f};
    float area_travel{0.0f};
    float lateral_travel{0.0f};
    float peak_bottom{0.0f};
    float peak_area{0.0f};
    float approach_peak_score{0.0f};
    float return_score{0.0f};
    int peak_index{-1};
    bool reversed_after_approach{false};
    bool reliable{false};
    int samples{0};
};

static Track create_track(const Detection& det, int id, bool already_captured = false);

static TrackSnapshot make_track_snapshot(const Track& t) {
    static constexpr size_t kSnapshotTrajectoryLimit = 160;
    TrackSnapshot snapshot;
    snapshot.id = t.id;
    snapshot.bbox = t.bbox;
    snapshot.smoothed_bbox = t.smoothed_bbox;
    snapshot.prop = t.prop;
    snapshot.missed = t.missed;
    snapshot.hits = t.hits;
    snapshot.confirmed = t.confirmed;
    snapshot.bbox_jitter = t.bbox_jitter;
    snapshot.is_approaching = t.is_approaching;
    snapshot.trajectory_direction = t.trajectory_direction;
    snapshot.trajectory_score = t.trajectory_score;
    snapshot.max_area_ratio = t.max_area_ratio;
    snapshot.has_reversed = t.has_reversed;
    snapshot.peak_bottom = t.peak_bottom;
    snapshot.peak_area = t.peak_area;
    snapshot.has_captured = t.has_captured;
    size_t path_size = t.trajectory_history.size();
    if (path_size <= kSnapshotTrajectoryLimit) {
        snapshot.trajectory_history = t.trajectory_history;
    } else {
        snapshot.trajectory_history.reserve(kSnapshotTrajectoryLimit);
        double step = static_cast<double>(path_size - 1) /
                      static_cast<double>(kSnapshotTrajectoryLimit - 1);
        for (size_t i = 0; i < kSnapshotTrajectoryLimit; ++i) {
            size_t index = std::min(path_size - 1, static_cast<size_t>(std::round(i * step)));
            snapshot.trajectory_history.push_back(t.trajectory_history[index]);
        }
    }

    for (auto it = t.frame_candidates.rbegin(); it != t.frame_candidates.rend(); ++it) {
        if (it->has_face && it->face_bbox_720p.width > 0 && it->face_bbox_720p.height > 0) {
            snapshot.has_face = true;
            snapshot.face_bbox_720p = it->face_bbox_720p;
            snapshot.face_confidence = it->face_confidence;
            break;
        }
    }
    return snapshot;
}

static std::vector<TrackSnapshot> make_track_snapshots(const std::vector<Track>& source) {
    std::vector<TrackSnapshot> snapshots;
    snapshots.reserve(source.size());
    for (const auto& t : source) {
        snapshots.push_back(make_track_snapshot(t));
    }
    return snapshots;
}

static bool is_valid_bbox(const cv::Rect2f& bbox) {
    return bbox.width > 1.0f && bbox.height > 1.0f;
}

static cv::Rect2f clamp_bbox(const cv::Rect2f& bbox) {
    float width = std::max(10.0f, std::min((float)IMAGE_WIDTH, bbox.width));
    float height = std::max(10.0f, std::min((float)IMAGE_HEIGHT, bbox.height));
    float x = std::max(0.0f, std::min((float)IMAGE_WIDTH - width, bbox.x));
    float y = std::max(0.0f, std::min((float)IMAGE_HEIGHT - height, bbox.y));
    return cv::Rect2f(x, y, width, height);
}

static cv::Rect2f blend_bbox(const cv::Rect2f& prev,
                             const cv::Rect2f& curr,
                             float center_alpha,
                             float size_alpha) {
    if (!is_valid_bbox(prev)) {
        return clamp_bbox(curr);
    }
    if (!is_valid_bbox(curr)) {
        return clamp_bbox(prev);
    }

    center_alpha = std::max(0.0f, std::min(1.0f, center_alpha));
    size_alpha = std::max(0.0f, std::min(1.0f, size_alpha));

    cv::Point2f prev_center(prev.x + prev.width * 0.5f, prev.y + prev.height * 0.5f);
    cv::Point2f curr_center(curr.x + curr.width * 0.5f, curr.y + curr.height * 0.5f);
    cv::Point2f center(prev_center.x * (1.0f - center_alpha) + curr_center.x * center_alpha,
                       prev_center.y * (1.0f - center_alpha) + curr_center.y * center_alpha);

    float width = prev.width * (1.0f - size_alpha) + curr.width * size_alpha;
    float height = prev.height * (1.0f - size_alpha) + curr.height * size_alpha;
    return clamp_bbox(cv::Rect2f(center.x - width * 0.5f,
                                 center.y - height * 0.5f,
                                 width,
                                 height));
}

static void append_area_history(std::vector<float>& history, float area) {
    history.push_back(std::max(1.0f, area));
    if (history.size() > BBOX_HISTORY_LIMIT) {
        history.erase(history.begin());
    }
}

static cv::Rect2f stable_track_bbox(const Track& t) {
    if (is_valid_bbox(t.smoothed_bbox)) {
        return t.smoothed_bbox;
    }
    return clamp_bbox(t.bbox);
}

static const char* track_trajectory_direction_to_string(TrackTrajectoryDirection direction) {
    switch (direction) {
        case TrackTrajectoryDirection::Approaching:
            return "approaching";
        case TrackTrajectoryDirection::Leaving:
            return "leaving";
        default:
            return "unknown";
    }
}

static float mean_range(const std::vector<float>& values, size_t begin, size_t end) {
    if (begin >= end || begin >= values.size()) {
        return 0.0f;
    }
    end = std::min(end, values.size());
    float sum = 0.0f;
    for (size_t i = begin; i < end; ++i) {
        sum += values[i];
    }
    return sum / static_cast<float>(end - begin);
}

static float compute_trajectory_delta_score(float bottom_from,
                                            float bottom_to,
                                            float area_from,
                                            float area_to) {
    float area_delta_ratio = (area_to - area_from) / std::max(area_from, 1e-4f);
    float bottom_component = (bottom_to - bottom_from) * 2.4f;
    float area_component = std::tanh(area_delta_ratio * 0.92f);
    return bottom_component * 0.68f + area_component * 0.32f;
}

static void append_track_trajectory_sample(Track& t, const cv::Rect2f& bbox) {
    cv::Point2f bottom_center(bbox.x + bbox.width * 0.5f,
                              bbox.y + bbox.height);
    t.trajectory_history.push_back(bottom_center);

    float area_ratio = bbox.area() /
        static_cast<float>(IMAGE_WIDTH * IMAGE_HEIGHT);
    t.trajectory_area_history.push_back(area_ratio);
    t.max_area_ratio = std::max(t.max_area_ratio, area_ratio);
}

static TrackTrajectoryDecision evaluate_track_trajectory(const Track& t) {
    TrackTrajectoryDecision decision;

    size_t sample_count = std::min(t.trajectory_history.size(), t.trajectory_area_history.size());
    if (sample_count < 5) {
        return decision;
    }

    const size_t start_index = 0;
    const size_t eval_count = sample_count;
    const size_t edge_window = std::min<size_t>(
        TRAJECTORY_EDGE_WINDOW_MAX,
        std::max<size_t>(2, eval_count / 6));

    std::vector<float> bottom_history;
    std::vector<float> area_history;
    bottom_history.reserve(eval_count);
    area_history.reserve(eval_count);

    const float inv_height = 1.0f / static_cast<float>(IMAGE_HEIGHT);
    const float inv_width = 1.0f / static_cast<float>(IMAGE_WIDTH);

    for (size_t i = start_index; i < sample_count; ++i) {
        bottom_history.push_back(t.trajectory_history[i].y * inv_height);
        area_history.push_back(t.trajectory_area_history[i]);
    }

    decision.samples = static_cast<int>(eval_count);
    decision.bottom_start = mean_range(bottom_history, 0, edge_window);
    decision.bottom_end = mean_range(bottom_history, eval_count - edge_window, eval_count);
    decision.area_start = mean_range(area_history, 0, edge_window);
    decision.area_end = mean_range(area_history, eval_count - edge_window, eval_count);
    decision.peak_bottom = bottom_history[0];
    decision.peak_area = area_history[0];
    decision.peak_index = 0;

    float bottom_signed = 0.0f;
    float area_signed = 0.0f;
    for (size_t i = 1; i < eval_count; ++i) {
        float bottom_step = bottom_history[i] - bottom_history[i - 1];
        float area_step = area_history[i] - area_history[i - 1];
        decision.bottom_travel += std::fabs(bottom_step);
        decision.area_travel += std::fabs(area_step);
        bottom_signed += bottom_step;
        area_signed += area_step;

        const cv::Point2f& prev = t.trajectory_history[start_index + i - 1];
        const cv::Point2f& curr = t.trajectory_history[start_index + i];
        decision.lateral_travel += std::fabs(curr.x - prev.x) * inv_width;
    }

    for (size_t i = 1; i < eval_count; ++i) {
        float area_rel = area_history[i] / std::max(g_capture_near_area_ratio, 1e-4f);
        float best_area_rel = decision.peak_area / std::max(g_capture_near_area_ratio, 1e-4f);
        float sample_proximity = bottom_history[i] * 0.68f + std::tanh(area_rel) * 0.32f;
        float peak_proximity = decision.peak_bottom * 0.68f + std::tanh(best_area_rel) * 0.32f;
        if (sample_proximity > peak_proximity) {
            decision.peak_bottom = bottom_history[i];
            decision.peak_area = area_history[i];
            decision.peak_index = static_cast<int>(i);
        }
    }

    if (decision.bottom_travel < TRAJECTORY_MIN_BOTTOM_TRAVEL &&
        decision.area_travel < TRAJECTORY_MIN_AREA_TRAVEL) {
        return decision;
    }

    float bottom_consistency =
        std::fabs(bottom_signed) / std::max(decision.bottom_travel, 1e-4f);
    float area_consistency =
        std::fabs(area_signed) / std::max(decision.area_travel, 1e-4f);
    float area_delta_ratio =
        (decision.area_end - decision.area_start) / std::max(decision.area_start, 1e-4f);

    decision.score = compute_trajectory_delta_score(decision.bottom_start,
                                                    decision.bottom_end,
                                                    decision.area_start,
                                                    decision.area_end);
    decision.approach_peak_score = compute_trajectory_delta_score(decision.bottom_start,
                                                                  decision.peak_bottom,
                                                                  decision.area_start,
                                                                  decision.peak_area);
    decision.return_score = compute_trajectory_delta_score(decision.bottom_end,
                                                           decision.peak_bottom,
                                                           decision.area_end,
                                                           decision.peak_area);

    bool mostly_sideways =
        decision.lateral_travel > decision.bottom_travel * 1.8f &&
        std::fabs(area_delta_ratio) < 0.18f;
    if (mostly_sideways) {
        decision.score *= 0.55f;
        decision.approach_peak_score *= 0.55f;
        decision.return_score *= 0.55f;
    }

    decision.reliable =
        bottom_consistency >= 0.35f ||
        area_consistency >= 0.45f ||
        std::fabs(area_delta_ratio) >= 0.45f ||
        std::fabs(decision.score) >= 0.2f; // 如果score绝对值较大，也认为可靠

    bool peak_before_end = decision.peak_index >= edge_window &&
                           decision.peak_index < static_cast<int>(eval_count - edge_window);
    float bottom_return_drop = decision.peak_bottom - decision.bottom_end;
    float area_return_drop =
        (decision.peak_area - decision.area_end) / std::max(decision.peak_area, 1e-4f);
    bool moved_back_from_near =
        decision.peak_bottom >= 0.72f &&
        decision.peak_index >= 0 &&
        decision.peak_index < static_cast<int>(eval_count - edge_window) &&
        ((bottom_return_drop >= 0.12f && decision.bottom_end < 0.72f) ||
         (area_return_drop >= 0.40f && decision.bottom_end < 0.65f));
    if (moved_back_from_near ||
        (decision.reliable &&
         peak_before_end &&
         decision.approach_peak_score >= TRAJECTORY_SCORE_THRESHOLD &&
         decision.return_score >= TRAJECTORY_RETURN_SCORE_THRESHOLD)) {
        decision.reversed_after_approach = true;
    }

    // 即使可靠性不高，只要score绝对值足够大，也设置方向
    if (decision.score >= TRAJECTORY_SCORE_THRESHOLD) {
        decision.direction = TrackTrajectoryDirection::Approaching;
    } else if (decision.score <= -TRAJECTORY_SCORE_THRESHOLD) {
        decision.direction = TrackTrajectoryDirection::Leaving;
    }

    return decision;
}

static void update_track_trajectory_state(Track& t) {
    TrackTrajectoryDecision decision = evaluate_track_trajectory(t);
    t.trajectory_direction = decision.direction;
    t.trajectory_score = decision.score;
    t.is_approaching = decision.direction == TrackTrajectoryDirection::Approaching;
    t.has_reversed = decision.reversed_after_approach;
    t.peak_bottom = decision.peak_bottom;
    t.peak_area = decision.peak_area;
    
    // Log trajectory direction analysis
    if (decision.samples >= 5) {
        const char* direction_str = track_trajectory_direction_to_string(decision.direction);
        log_debug("Track %d trajectory analysis: direction=%s score=%.3f bottom=%.3f->%.3f area=%.4f->%.4f",
                 t.id,
                 direction_str,
                 decision.score,
                 decision.bottom_start,
                 decision.bottom_end,
                 decision.area_start,
                 decision.area_end);
    }
}

static bool should_upload_track_by_trajectory(const Track& t,
                                              TrackTrajectoryDecision* out_decision,
                                              const char** reason) {
    TrackTrajectoryDecision decision = evaluate_track_trajectory(t);
    if (out_decision) {
        *out_decision = decision;
    }

    float bottom_position = 0.0f;
    if (!t.trajectory_history.empty()) {
        bottom_position = t.trajectory_history.back().y / static_cast<float>(IMAGE_HEIGHT);
    } else {
        bottom_position = (t.bbox.y + t.bbox.height) / static_cast<float>(IMAGE_HEIGHT);
    }

    float current_area = decision.area_end;
    if (current_area <= 0.0f && !t.trajectory_area_history.empty()) {
        current_area = t.trajectory_area_history.back();
    }
    float max_area = std::max(decision.peak_area, t.max_area_ratio);
    bool reached_near =
        decision.peak_bottom >= 0.75f ||
        max_area >= g_capture_near_area_ratio;
    bool ended_near_bottom = bottom_position >= 0.68f;
    bool started_near =
        decision.bottom_start >= 0.70f ||
        decision.area_start >= g_capture_near_area_ratio * 0.90f;
    bool approach_seen =
        decision.direction == TrackTrajectoryDirection::Approaching ||
        decision.approach_peak_score >= 0.06f;

    if (decision.reversed_after_approach) {
        if (reason) {
            *reason = "trajectory_returned_after_near";
        }
        return false;
    }

    if (reached_near && bottom_position < 0.50f) {
        if (reason) {
            *reason = "trajectory_returned_to_far";
        }
        return false;
    }

    if (g_capture_require_approach && !approach_seen && !started_near) {
        if (reason) {
            *reason = "no_full_path_approach";
        }
        return false;
    }

    bool significantly_smaller =
        max_area > 0.0f &&
        current_area > 0.0f &&
        current_area < max_area * 0.55f;
    bool area_shrinking_near_exit =
        significantly_smaller &&
        reached_near &&
        bottom_position >= 0.58f;

    if (t.missed > 1 && reached_near && (ended_near_bottom || area_shrinking_near_exit)) {
        if (reason) {
            *reason = ended_near_bottom ? "trajectory_lost_near_bottom" : "trajectory_area_shrunk_near_bottom";
        }
        return true;
    }

    if (reason) {
        *reason = "not_lost_near_bottom";
    }
    return false;
}

void sort_init() { 
    std::unique_lock<std::mutex> lock(tracks_mutex);
    tracks.clear(); 
    lost_tracks.clear();
    recent_captures.clear();
    pending_tracks.clear();
    next_id = 1; 
}

// 娣诲姞涓婁紶鍥炶皟鍑芥暟鎸囬拡
static std::function<void(const cv::Mat&, int, const std::string&)> upload_callback = nullptr;
static std::unordered_set<int>* captured_person_ids = nullptr;
static std::unordered_set<int>* captured_face_ids = nullptr;

void set_upload_callback(std::function<void(const cv::Mat&, int, const std::string&)> callback,
                        std::unordered_set<int>* person_ids, std::unordered_set<int>* face_ids) {
    upload_callback = callback;
    captured_person_ids = person_ids;
    captured_face_ids = face_ids;
}

void set_max_frame_candidates(size_t maxFrameCandidates) {
    std::lock_guard<std::mutex> lock(tracks_mutex);
    g_max_frame_candidates = std::max<size_t>(1, maxFrameCandidates);
}

void set_capture_sort_preferences(float minAreaRatio,
                                  float nearAreaRatio,
                                  float maxPersonOcclusion,
                                  bool requireApproach) {
    std::lock_guard<std::mutex> lock(tracks_mutex);
    g_capture_min_area_ratio = std::max(0.001f, minAreaRatio);
    g_capture_near_area_ratio = std::max(g_capture_min_area_ratio, nearAreaRatio);
    g_capture_max_person_occlusion = std::max(0.05f, maxPersonOcclusion);
    g_capture_require_approach = requireApproach;
}

static bool is_track_person_captured(int track_id) {
    return captured_person_ids &&
           captured_person_ids->find(track_id) != captured_person_ids->end();
}

static bool is_track_face_captured(int track_id) {
    return captured_face_ids &&
           captured_face_ids->find(track_id) != captured_face_ids->end();
}

static bool is_track_fully_captured(int track_id) {
    return is_track_person_captured(track_id) && is_track_face_captured(track_id);
}

static bool has_track_uploaded_asset(int track_id) {
    return is_track_person_captured(track_id) || is_track_face_captured(track_id);
}

static float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    float xx1 = std::max(a.x, b.x);
    float yy1 = std::max(a.y, b.y);
    float xx2 = std::min(a.x + a.width, b.x + b.width);
    float yy2 = std::min(a.y + a.height, b.y + b.height);
    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter = w * h;
    return inter / (a.area() + b.area() - inter + 1e-6f);
}

static cv::Mat calc_hist(const cv::Mat& roi) {
    cv::Mat hsv;
    cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);
    int h_bins = 16;
    int s_bins = 16;
    int histSize[] = {h_bins, s_bins};
    float h_ranges[] = {0,180};
    float s_ranges[] = {0,256};
    const float* ranges[] = {h_ranges, s_ranges};
    int channels[] = {0,1};
    cv::Mat hist;
    cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 2, histSize, ranges);
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);
    return hist;
}

static float hist_distance(const cv::Mat& a, const cv::Mat& b) {
    return cv::compareHist(a, b, cv::HISTCMP_BHATTACHARYYA);
}

static float center_distance_norm(const cv::Rect2f& a, const cv::Rect2f& b) {
    float ax = a.x + a.width * 0.5f;
    float ay = a.y + a.height * 0.5f;
    float bx = b.x + b.width * 0.5f;
    float by = b.y + b.height * 0.5f;
    float dx = ax - bx;
    float dy = ay - by;
    float diag = std::sqrt((float)IMAGE_WIDTH * IMAGE_WIDTH + (float)IMAGE_HEIGHT * IMAGE_HEIGHT);
    return std::sqrt(dx * dx + dy * dy) / (diag + 1e-6f);
}

static void age_pending_tracks() {
    for (auto& pt : pending_tracks) {
        pt.ttl--;
    }
    pending_tracks.erase(
        std::remove_if(pending_tracks.begin(), pending_tracks.end(), [](const PendingTrack& pt) {
            return pt.ttl <= 0;
        }),
        pending_tracks.end());
}

static void age_lost_tracks() {
    for (auto& lt : lost_tracks) {
        lt.ttl--;
    }
    lost_tracks.erase(
        std::remove_if(lost_tracks.begin(), lost_tracks.end(), [](const LostTrack& lt) {
            return lt.ttl <= 0;
        }),
        lost_tracks.end());
}

static void age_recent_captures() {
    for (auto& rc : recent_captures) {
        rc.ttl--;
    }
    recent_captures.erase(
        std::remove_if(recent_captures.begin(), recent_captures.end(), [](const RecentCapture& rc) {
            return rc.ttl <= 0;
        }),
        recent_captures.end());
}

static void cache_lost_track(const Track& t) {
    if (!t.confirmed || t.hist.empty()) {
        return;
    }

    lost_tracks.erase(
        std::remove_if(lost_tracks.begin(), lost_tracks.end(), [&](const LostTrack& lt) {
            return lt.id == t.id;
        }),
        lost_tracks.end());

    LostTrack lt;
    lt.id = t.id;
    lt.bbox = stable_track_bbox(t);
    lt.hist = t.hist.clone();
    lt.ttl = LOST_TRACK_TTL;
    lost_tracks.push_back(std::move(lt));
}

static void remember_recent_capture(const Track& t) {
    if (!has_track_uploaded_asset(t.id) || t.hist.empty()) {
        return;
    }

    recent_captures.erase(
        std::remove_if(recent_captures.begin(), recent_captures.end(), [&](const RecentCapture& rc) {
            return rc.id == t.id;
        }),
        recent_captures.end());

    RecentCapture rc;
    rc.id = t.id;
    rc.bbox = stable_track_bbox(t);
    rc.hist = t.hist.clone();
    rc.ttl = RECENT_CAPTURE_TTL;
    recent_captures.push_back(std::move(rc));
}

static int reuse_lost_track_id(const Detection& det, const cv::Mat& det_hist) {
    if (lost_tracks.empty() || det_hist.empty()) {
        return -1;
    }

    cv::Rect2f det_rect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);
    float best_score = -1.0f;
    int best_idx = -1;

    for (int i = 0; i < static_cast<int>(lost_tracks.size()); ++i) {
        const auto& lt = lost_tracks[i];
        if (lt.hist.empty()) {
            continue;
        }

        float iou_score = iou(lt.bbox, det_rect);
        float hist_score = hist_distance(lt.hist, det_hist);
        float center_dist = center_distance_norm(lt.bbox, det_rect);
        float center_score = 1.0f - std::min(1.0f, center_dist / 0.20f);

        if ((iou_score < 0.03f && center_dist > 0.15f) || hist_score > 0.50f || center_score < 0.0f) {
            continue;
        }

        float score = iou_score * 0.45f + (1.0f - hist_score) * 0.35f + center_score * 0.20f;
        // TTL decay: older lost tracks are harder to reuse.
        float ttl_factor = static_cast<float>(lt.ttl) / static_cast<float>(LOST_TRACK_TTL);
        score *= std::max(0.3f, ttl_factor);
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    if (best_idx != -1 && best_score >= 0.55f) {
        int reused_id = lost_tracks[best_idx].id;
        lost_tracks.erase(lost_tracks.begin() + best_idx);
        log_debug("Reuse lost track id: %d (score=%.3f)", reused_id, best_score);
        return reused_id;
    }

    return -1;
}

static int reuse_recent_capture_id(const Detection& det, const cv::Mat& det_hist) {
    if (recent_captures.empty() || det_hist.empty()) {
        return -1;
    }

    cv::Rect2f det_rect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);
    float best_score = -1.0f;
    int best_idx = -1;

    for (int i = 0; i < static_cast<int>(recent_captures.size()); ++i) {
        const auto& rc = recent_captures[i];
        if (rc.hist.empty()) {
            continue;
        }

        float hist_score = hist_distance(rc.hist, det_hist);
        float iou_score = iou(rc.bbox, det_rect);
        float center_dist = center_distance_norm(rc.bbox, det_rect);
        float center_score = 1.0f - std::min(1.0f, center_dist / 0.14f);

        if (hist_score > 0.28f) {
            continue;
        }
        if (iou_score < 0.04f && center_dist > 0.08f) {
            continue;
        }

        float score = (1.0f - hist_score) * 0.55f + center_score * 0.30f + iou_score * 0.15f;
        // TTL decay: older captures are harder to reuse.
        float ttl_factor = static_cast<float>(rc.ttl) / static_cast<float>(RECENT_CAPTURE_TTL);
        score *= std::max(0.4f, ttl_factor);
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    if (best_idx != -1 && best_score >= 0.75f) {
        int reused_id = recent_captures[best_idx].id;
        recent_captures.erase(recent_captures.begin() + best_idx);
        log_debug("Reuse recent captured id: %d (score=%.3f)", reused_id, best_score);
        return reused_id;
    }

    return -1;
}

static bool is_usable_face_frame(const Track::FrameData& frame) {
    return frame.has_face && frame.face_pose_level >= 1 && !frame.face_roi.empty();
}

static float frame_occlusion(const Track::FrameData& frame) {
    return frame.person_occlusion * 0.6f + frame.face_edge_occlusion * 0.4f;
}

static float near_ratio_score(const Track::FrameData& frame) {
    return frame.area_ratio / std::max(1e-6f, g_capture_near_area_ratio);
}

static double face_capture_priority(const Track::FrameData& frame) {
    float occ = frame_occlusion(frame);
    float near_ratio = near_ratio_score(frame);
    float pose_bonus = frame.face_pose_level >= 2 ? 150.0f : 82.0f;
    float strong_bonus = frame.strong_candidate ? 95.0f : 0.0f;
    float clarity_bonus = static_cast<float>(frame.clarity * 0.46);
    float near_bonus = std::min(2.5f, near_ratio) * 280.0f;
    float far_penalty = near_ratio < 1.0f ? (1.0f - near_ratio) * 380.0f : 0.0f;
    float blur_penalty = frame.blur_severity * 95.0f;
    float motion_penalty = frame.motion_ratio * 24000.0f;
    float occ_penalty = frame.person_occlusion * 120.0f + frame.face_edge_occlusion * 150.0f;
    float yaw_penalty = frame.yaw_abs * 72.0f;
    return frame.score + pose_bonus + strong_bonus + clarity_bonus + near_bonus -
           blur_penalty - motion_penalty - occ_penalty - yaw_penalty - far_penalty -
           occ * 25.0f;
}

static double person_capture_priority(const Track::FrameData& frame) {
    float near_ratio = near_ratio_score(frame);
    float near_bonus = std::min(2.5f, near_ratio) * 320.0f;
    float far_penalty = near_ratio < 1.0f ? (1.0f - near_ratio) * 450.0f : 0.0f;
    float clarity_bonus = static_cast<float>(frame.clarity * 0.34);
    float blur_penalty = frame.blur_severity * 88.0f;
    float motion_penalty = frame.motion_ratio * 20000.0f;
    float occ_penalty = frame.person_occlusion * 165.0f;
    float face_hint_bonus = is_usable_face_frame(frame) ? 35.0f : 0.0f;
    return frame.score + near_bonus + clarity_bonus + face_hint_bonus -
           blur_penalty - motion_penalty - occ_penalty - far_penalty;
}

static double overall_capture_priority(const Track::FrameData& frame) {
    return is_usable_face_frame(frame)
        ? face_capture_priority(frame)
        : person_capture_priority(frame) - 260.0;
}

static bool better_face_capture(const Track::FrameData& candidate,
                                const Track::FrameData& current) {
    if (!is_usable_face_frame(candidate)) {
        return false;
    }
    if (!is_usable_face_frame(current)) {
        return true;
    }

    float candidate_occ = frame_occlusion(candidate);
    float current_occ = frame_occlusion(current);
    double candidate_priority = face_capture_priority(candidate);
    double current_priority = face_capture_priority(current);

    bool much_clearer = candidate.clarity > current.clarity * 1.15 + 4.0 &&
                        candidate.blur_severity <= current.blur_severity + 0.08f &&
                        candidate.motion_ratio <= current.motion_ratio + 0.004f &&
                        candidate_occ <= current_occ + 0.10f;
    if (much_clearer) {
        return true;
    }

    if (candidate.face_pose_level > current.face_pose_level &&
        candidate.clarity >= current.clarity * 0.86 &&
        candidate.blur_severity <= current.blur_severity + 0.10f &&
        candidate.area_ratio >= current.area_ratio * 0.92f) {
        return true;
    }

    if (candidate.strong_candidate != current.strong_candidate) {
        if (candidate.strong_candidate &&
            candidate.clarity >= current.clarity * 0.90 &&
            candidate.blur_severity <= current.blur_severity + 0.08f) {
            return true;
        }
        if (!candidate.strong_candidate &&
            current.strong_candidate &&
            current.clarity >= candidate.clarity * 0.90 &&
            current.blur_severity <= candidate.blur_severity + 0.08f) {
            return false;
        }
    }

    if (candidate_priority > current_priority + 18.0) {
        return true;
    }
    if (candidate_priority < current_priority - 18.0) {
        return false;
    }

    if (candidate_occ + 0.02f < current_occ &&
        candidate.clarity >= current.clarity * 0.96f &&
        candidate.area_ratio >= current.area_ratio * 0.92f) {
        return true;
    }

    if (candidate.area_ratio > current.area_ratio * 1.08f &&
        candidate.blur_severity <= current.blur_severity + 0.06f &&
        candidate_occ <= current_occ + 0.08f) {
        return true;
    }

    if (candidate.blur_severity + 0.04f < current.blur_severity &&
        candidate.clarity >= current.clarity * 0.96f) {
        return true;
    }

    if (candidate.motion_ratio + 0.0025f < current.motion_ratio &&
        candidate.clarity >= current.clarity * 0.97f) {
        return true;
    }

    if (candidate.yaw_abs + 0.03f < current.yaw_abs &&
        candidate.clarity >= current.clarity * 0.98f &&
        candidate.blur_severity <= current.blur_severity + 0.03f) {
        return true;
    }

    return candidate.score > current.score &&
           candidate.clarity >= current.clarity * 0.98;
}

static bool better_person_capture(const Track::FrameData& candidate,
                                  const Track::FrameData& current) {
    if (candidate.person_roi.empty()) {
        return false;
    }
    if (current.person_roi.empty()) {
        return true;
    }

    double candidate_priority = person_capture_priority(candidate);
    double current_priority = person_capture_priority(current);

    if (candidate.area_ratio > current.area_ratio * 1.12f &&
        candidate.person_occlusion <= current.person_occlusion + 0.10f &&
        candidate.blur_severity <= current.blur_severity + 0.08f) {
        return true;
    }

    if (candidate.person_occlusion + 0.08f < current.person_occlusion &&
        candidate.area_ratio >= current.area_ratio * 0.94f &&
        candidate.clarity >= current.clarity * 0.92f) {
        return true;
    }

    if (candidate_priority > current_priority + 16.0) {
        return true;
    }
    if (candidate_priority < current_priority - 16.0) {
        return false;
    }

    if (candidate.clarity > current.clarity * 1.08f + 3.0 &&
        candidate.blur_severity <= current.blur_severity + 0.06f) {
        return true;
    }

    return candidate.score > current.score &&
           candidate.area_ratio >= current.area_ratio * 0.96f;
}

static size_t select_best_face_frame_index(const std::vector<Track::FrameData>& frames) {
    size_t best_index = SIZE_MAX;
    for (size_t i = 0; i < frames.size(); ++i) {
        if (!is_usable_face_frame(frames[i])) {
            continue;
        }
        if (best_index == SIZE_MAX || better_face_capture(frames[i], frames[best_index])) {
            best_index = i;
        }
    }
    return best_index;
}

static size_t select_best_person_frame_index(const std::vector<Track::FrameData>& frames) {
    size_t best_index = SIZE_MAX;
    for (size_t i = 0; i < frames.size(); ++i) {
        if (frames[i].person_roi.empty()) {
            continue;
        }
        if (best_index == SIZE_MAX || better_person_capture(frames[i], frames[best_index])) {
            best_index = i;
        }
    }
    return best_index;
}

static bool should_suppress_new_track(const Detection& det) {
    cv::Rect2f det_rect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);

    // Check against active tracks.
    for (const auto& t : tracks) {
        cv::Rect2f ref_bbox = stable_track_bbox(t);
        float iou_score = iou(ref_bbox, det_rect);
        float center_dist = center_distance_norm(ref_bbox, det_rect);
        if (iou_score > 0.38f || center_dist < 0.05f) {
            return true;
        }
    }
    // Also check against pending tracks to avoid duplicate pending entries.
    for (const auto& pt : pending_tracks) {
        float iou_score = iou(pt.bbox, det_rect);
        float center_dist = center_distance_norm(pt.bbox, det_rect);
        if (iou_score > 0.38f || center_dist < 0.05f) {
            return true;
        }
    }
    return false;
}

static int update_pending_track(const Detection& det, const cv::Mat& det_hist) {
    cv::Rect2f det_rect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);

    int best_idx = -1;
    float best_score = -1.0f;
    for (int i = 0; i < static_cast<int>(pending_tracks.size()); ++i) {
        float iou_score = iou(pending_tracks[i].bbox, det_rect);
        float hist_score = hist_distance(pending_tracks[i].hist, det_hist);
        float center_dist = center_distance_norm(pending_tracks[i].bbox, det_rect);
        float center_score = 1.0f - std::min(1.0f, center_dist / 0.15f);
        float score = iou_score * 0.45f + (1.0f - hist_score) * 0.35f + center_score * 0.20f;
        if ((iou_score > 0.12f || center_dist < 0.05f) && hist_score < 0.55f && score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    if (best_idx == -1) {
        PendingTrack pt;
        pt.bbox = det_rect;
        pt.hist = det_hist.clone();
        pt.prop = det.prop;
        pt.hits = 1;
        pt.ttl = PENDING_TRACK_TTL;
        pending_tracks.push_back(std::move(pt));
        return -1;
    }

    auto& pt = pending_tracks[best_idx];
    pt.bbox = blend_bbox(pt.bbox, det_rect, 0.55f, 0.35f);
    pt.hist = det_hist.clone();
    pt.prop = det.prop;
    pt.hits++;
    pt.ttl = PENDING_TRACK_TTL;

    if (pt.hits >= PENDING_TRACK_HITS_REQUIRED) {
        int reused_id = reuse_lost_track_id(det, det_hist);
        if (reused_id <= 0) {
            reused_id = reuse_recent_capture_id(det, det_hist);
        }
        int assigned_id = (reused_id > 0) ? reused_id : next_id++;
        bool already_captured = is_track_fully_captured(assigned_id);

        tracks.push_back(create_track(det, assigned_id, already_captured));
        pending_tracks.erase(pending_tracks.begin() + best_idx);
        return assigned_id;
    }

    return -1;
}


static std::vector<std::pair<int,int>> hungarian_algorithm(const std::vector<std::vector<float>>& cost_matrix, float max_cost) {
    // Kuhn-Munkres (Hungarian) algorithm — O(n^3) globally optimal assignment.
    // For n<10 (typical 1-5 persons) this runs in <0.1ms on ARM Cortex-A55.
    int n = static_cast<int>(cost_matrix.size());
    int m = n > 0 ? static_cast<int>(cost_matrix[0].size()) : 0;
    if (n == 0 || m == 0) return {};

    int sz = std::max(n, m);
    std::vector<std::vector<float>> C(sz, std::vector<float>(sz, max_cost));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            C[i][j] = cost_matrix[i][j];

    const float INF = 1e9f;
    std::vector<float> u(sz + 1, 0.0f), v(sz + 1, 0.0f);
    std::vector<int> p(sz + 1, 0), way(sz + 1, 0);

    for (int i = 1; i <= sz; i++) {
        p[0] = i;
        int j0 = 0;
        std::vector<float> minv(sz + 1, INF);
        std::vector<bool> used(sz + 1, false);
        do {
            used[j0] = true;
            int i0 = p[j0], j1 = 0;
            float delta = INF;
            for (int j = 1; j <= sz; j++) {
                if (!used[j]) {
                    float cur = C[i0 - 1][j - 1] - u[i0] - v[j];
                    if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                    if (minv[j] < delta) { delta = minv[j]; j1 = j; }
                }
            }
            for (int j = 0; j <= sz; j++) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else { minv[j] -= delta; }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do { int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
    }

    std::vector<std::pair<int,int>> assignments;
    for (int j = 1; j <= sz; j++) {
        int i = p[j] - 1;
        int col = j - 1;
        if (i >= 0 && i < n && col >= 0 && col < m && cost_matrix[i][col] < max_cost) {
            assignments.push_back({i, col});
        }
    }
    return assignments;
}

//-----------------Track EKF鎿嶄綔-----------------

static void predict_track(Track& t) {
    // 鏀硅繘鐨?鐘舵€佽繍鍔ㄦā鍨? [x, y, w, h, vx, vy, vw, vh]
    _float_t dt = 1.0f;  // 鏃堕棿姝ラ暱
    _float_t F[EKF_N*EKF_N] = {
        1,0,0,0,dt, 0, 0, 0,  // x = x + vx*dt
        0,1,0,0, 0,dt, 0, 0,  // y = y + vy*dt  
        0,0,1,0, 0, 0,dt, 0,  // w = w + vw*dt
        0,0,0,1, 0, 0, 0,dt,  // h = h + vh*dt
        0,0,0,0, 1, 0, 0, 0,  // vx = vx (甯搁€熷害妯″瀷)
        0,0,0,0, 0, 1, 0, 0,  // vy = vy
        0,0,0,0, 0, 0, 1, 0,  // vw = vw  
        0,0,0,0, 0, 0, 0, 1   // vh = vh
    };
    
    // 鏀硅繘鐨勮繃绋嬪櫔澹扮煩闃?- 鏍规嵁杩愬姩涓嶇‘瀹氭€ц皟鏁?
    _float_t Q[EKF_N*EKF_N] = {0};
    // 浣嶇疆鍣０
    Q[0*EKF_N+0] = 1.0f;   // x浣嶇疆鍣０
    Q[1*EKF_N+1] = 1.0f;   // y浣嶇疆鍣０
    Q[2*EKF_N+2] = 0.5f;   // width鍣０(杈冨皬)
    Q[3*EKF_N+3] = 0.5f;   // height鍣０(杈冨皬)
    // 閫熷害鍣０
    Q[4*EKF_N+4] = 0.1f;   // x閫熷害鍣０
    Q[5*EKF_N+5] = 0.1f;   // y閫熷害鍣０  
    Q[6*EKF_N+6] = 0.05f;  // width鍙樺寲閫熷害鍣０
    Q[7*EKF_N+7] = 0.05f;  // height鍙樺寲閫熷害鍣０
    
    ekf_predict(&t.ekf, t.ekf.x, F, Q);

    t.bbox = clamp_bbox(cv::Rect2f(t.ekf.x[0],
                                   t.ekf.x[1],
                                   std::max(10.0f, t.ekf.x[2]),
                                   std::max(10.0f, t.ekf.x[3])));
    t.bbox.width  = std::max(10.0f, t.ekf.x[2]);   // 闃叉瀹藉害杩囧皬
    t.bbox.height = std::max(10.0f, t.ekf.x[3]);   // 闃叉楂樺害杩囧皬

    // 杈圭晫妫€鏌ワ紝闃叉bbox瓒呭嚭鍥惧儚杈圭晫锛堢幇鍦ㄤ娇鐢?20p鍧愭爣绯伙級
    t.bbox.x = std::max(0.0f, std::min((float)(IMAGE_WIDTH - t.bbox.width), t.bbox.x));
    t.bbox.y = std::max(0.0f, std::min((float)(IMAGE_HEIGHT - t.bbox.height), t.bbox.y));
    
    // 纭繚bbox鍦ㄥ浘鍍忚寖鍥村唴
    if (t.bbox.x + t.bbox.width > IMAGE_WIDTH) {
        t.bbox.width = IMAGE_WIDTH - t.bbox.x;
    }
    if (t.bbox.y + t.bbox.height > IMAGE_HEIGHT) {
        t.bbox.height = IMAGE_HEIGHT - t.bbox.y;
    }

    t.age++;
    t.missed++;
}

static void correct_track(Track& t, const Detection& det) {
    _float_t z[EKF_M] = {det.x1, det.y1, det.x2-det.x1, det.y2-det.y1};
    _float_t H[EKF_M*EKF_N] = {1,0,0,0,
                                0,1,0,0,
                                0,0,1,0,
                                0,0,0,1};
    _float_t R[EKF_M*EKF_M] = {16.0,0,0,0,
                                0,16.0,0,0,
                                0,0,64.0,0,
                                0,0,0,64.0};
    ekf_update(&t.ekf, z, z, H, R);

    t.bbox = clamp_bbox(cv::Rect2f(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1));
    t.smoothed_bbox = t.bbox;
    t.last_det_bbox = t.bbox;
    t.hist = calc_hist(det.roi);
    t.prop = det.prop;  // 鏇存柊缃俊搴?
    t.missed = 0;
    t.hits++;
    t.active = true;
    
    // 缁忚繃3娆℃垚鍔熷尮閰嶅悗纭track
    if (t.hits >= 3) {
        t.confirmed = true;
    }

    // 璁板綍妫€娴嬫闈㈢Н鍘嗗彶
    float area = t.bbox.width * t.bbox.height;
    t.bbox_history.push_back(area);
    // 鍙繚鐣欐渶杩?0甯?
    if (t.bbox_history.size() > 20) t.bbox_history.erase(t.bbox_history.begin());
}

//-----------------鏂板缓Track-----------------

static void predict_track_robust(Track& t, bool count_missed = true) {
    _float_t dt = 1.0f;
    _float_t F[EKF_N * EKF_N] = {
        1,0,0,0,dt, 0, 0, 0,
        0,1,0,0, 0,dt, 0, 0,
        0,0,1,0, 0, 0,dt, 0,
        0,0,0,1, 0, 0, 0,dt,
        0,0,0,0, 1, 0, 0, 0,
        0,0,0,0, 0, 1, 0, 0,
        0,0,0,0, 0, 0, 1, 0,
        0,0,0,0, 0, 0, 0, 1
    };

    _float_t Q[EKF_N * EKF_N] = {0};
    Q[0 * EKF_N + 0] = 1.5f; //ofc is 1.0
    Q[1 * EKF_N + 1] = 1.5f; //ofc is 1.0
    Q[2 * EKF_N + 2] = 0.5f;
    Q[3 * EKF_N + 3] = 0.5f;
    Q[4 * EKF_N + 4] = 0.15f; //ofc is 0.1
    Q[5 * EKF_N + 5] = 0.15f; //ofc is 0.1
    Q[6 * EKF_N + 6] = 0.05f;
    Q[7 * EKF_N + 7] = 0.05f;

    ekf_predict(&t.ekf, t.ekf.x, F, Q);
    t.bbox = clamp_bbox(cv::Rect2f(t.ekf.x[0],
                                   t.ekf.x[1],
                                   std::max(10.0f, t.ekf.x[2]),
                                   std::max(10.0f, t.ekf.x[3])));
    t.age++;
    if (count_missed) {
        t.missed++;
    }
}

static void correct_track_robust(Track& t, const Detection& det) {
    int prev_hits = t.hits;
    int prev_missed = t.missed;
    cv::Rect2f prev_smoothed = t.smoothed_bbox;
    cv::Rect2f det_rect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);

    _float_t z[EKF_M] = {det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1};
    _float_t H[EKF_M * EKF_N] = {1,0,0,0,
                                 0,1,0,0,
                                 0,0,1,0,
                                 0,0,0,1};
    _float_t R[EKF_M * EKF_M] = {16.0,0,0,0,
                                 0,16.0,0,0,
                                 0,0,64.0,0,
                                 0,0,0,64.0};
    ekf_update(&t.ekf, z, z, H, R);

    cv::Rect2f ekf_rect = clamp_bbox(cv::Rect2f(t.ekf.x[0],
                                                t.ekf.x[1],
                                                std::max(10.0f, t.ekf.x[2]),
                                                std::max(10.0f, t.ekf.x[3])));
    cv::Rect2f fused_measurement = blend_bbox(det_rect, ekf_rect, 0.28f, 0.20f);

    float jitter_sample = 0.0f;
    if (is_valid_bbox(prev_smoothed)) {
        float width_jitter = std::fabs(det_rect.width - prev_smoothed.width) / (prev_smoothed.width + 1e-6f);
        float height_jitter = std::fabs(det_rect.height - prev_smoothed.height) / (prev_smoothed.height + 1e-6f);
        float area_jitter = std::fabs(det_rect.area() - prev_smoothed.area()) / (prev_smoothed.area() + 1e-6f);
        jitter_sample = std::min(1.0f, width_jitter * 0.35f + height_jitter * 0.35f + area_jitter * 0.30f);
    }

    float center_alpha = TRACK_SMOOTH_CENTER_ALPHA;
    float size_alpha = TRACK_SMOOTH_SIZE_ALPHA;
    if (!is_valid_bbox(prev_smoothed) || prev_hits < 2) {
        center_alpha = TRACK_NEW_CENTER_ALPHA;
        size_alpha = TRACK_NEW_SIZE_ALPHA;
    } else if (prev_missed > 1) {
        center_alpha = TRACK_RECOVER_CENTER_ALPHA;
        size_alpha = TRACK_RECOVER_SIZE_ALPHA;
    } else if (iou(prev_smoothed, det_rect) < 0.18f) {
        center_alpha = std::max(center_alpha, 0.55f);
        size_alpha = std::max(size_alpha, 0.36f);
    }

    t.last_det_bbox = det_rect;
    t.smoothed_bbox = is_valid_bbox(prev_smoothed)
        ? blend_bbox(prev_smoothed, fused_measurement, center_alpha, size_alpha)
        : clamp_bbox(fused_measurement);
    t.bbox = t.smoothed_bbox;
    t.hist = calc_hist(det.roi);
    t.prop = det.prop;
    t.missed = 0;
    t.hits++;
    t.active = true;
    t.bbox_jitter = t.bbox_jitter * (1.0f - TRACK_BBOX_JITTER_ALPHA) + jitter_sample * TRACK_BBOX_JITTER_ALPHA;

    if (t.hits >= 3) {
        t.confirmed = true;
    }

    append_area_history(t.bbox_history, t.bbox.area());
    append_track_trajectory_sample(t, t.bbox);
    update_track_trajectory_state(t);
}

static Track create_track(const Detection& det, int id, bool already_captured) {
    Track t;
    t.id = id;

    _float_t Pdiag[EKF_N] = {1,1,1,1,10,10,10,10};  // 浣嶇疆鏂瑰樊杈冨皬锛岄€熷害鏂瑰樊杈冨ぇ
    ekf_initialize(&t.ekf, Pdiag);

    _float_t state[EKF_N] = {det.x1, det.y1, det.x2-det.x1, det.y2-det.y1, 0, 0, 0, 0};
    memcpy(t.ekf.x, state, sizeof(state));

    t.bbox = clamp_bbox(cv::Rect2f(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1));
    t.smoothed_bbox = t.bbox;
    t.last_det_bbox = t.bbox;
    t.hist = calc_hist(det.roi);
    t.prop = det.prop;
    t.age = 1;
    t.missed = 0;
    t.hits = 1;  // 鍒濆鍛戒腑娆℃暟
    t.active = true;
    t.confirmed = false;  // 闇€瑕佸嚑甯х‘璁?
    append_area_history(t.bbox_history, t.bbox.area());
    t.bbox_jitter = 0.0f;
    t.is_approaching = false;
    t.trajectory_direction = TrackTrajectoryDirection::Unknown;
    t.trajectory_score = 0.0f;
    t.max_area_ratio = 0.0f;
    t.best_clarity = 0.0;
    t.has_captured = already_captured;
    t.has_reversed = false;
    t.peak_bottom = 0.0f;
    t.peak_area = 0.0f;
    append_track_trajectory_sample(t, t.bbox);
    update_track_trajectory_state(t);
    return t;
}

//-----------------涓绘洿鏂板嚱鏁?----------------

std::vector<TrackSnapshot> sort_update(const std::vector<Detection>& dets) {
    struct PendingUpload {
        int trackId;
        bool uploadPerson{false};
        bool uploadFace{false};
        Track::FrameData personFrame;
        Track::FrameData faceFrame;
        float faceOcclusion{0.0f};
    };

    std::vector<PendingUpload> pendingUploads;
    std::vector<TrackSnapshot> snapshot;

    std::unique_lock<std::mutex> lock(tracks_mutex);
    age_lost_tracks();
    age_recent_captures();
    age_pending_tracks();

    auto queue_upload_if_needed = [&](const Track& t) {
        if (!upload_callback || t.frame_candidates.empty() || !captured_person_ids || !captured_face_ids) {
            log_debug("Track %d upload conditions not met", t.id);
            if (has_track_uploaded_asset(t.id)) {
                remember_recent_capture(t);
            }
            return;
        }

        if (is_track_fully_captured(t.id)) {
            remember_recent_capture(t);
            log_debug("Track %d already captured, refresh recent cache only", t.id);
            return;
        }

        size_t best_face_index = select_best_face_frame_index(t.frame_candidates);
        size_t best_person_index = select_best_person_frame_index(t.frame_candidates);
        if (best_face_index == SIZE_MAX && best_person_index == SIZE_MAX) {
            log_debug("Track %d skipped upload: no usable candidate", t.id);
            return;
        }

        const float person_area_threshold =
            std::max(g_capture_min_area_ratio * 1.10f, g_capture_near_area_ratio * 0.82f);
        const float face_area_threshold =
            std::max(g_capture_min_area_ratio * 1.22f, g_capture_near_area_ratio * 0.88f);

        PendingUpload pending;
        pending.trackId = t.id;

        if (!is_track_face_captured(t.id) && best_face_index != SIZE_MAX) {
            const auto& best_face_frame = t.frame_candidates[best_face_index];
            if (best_face_frame.area_ratio >= face_area_threshold) {
                pending.uploadFace = true;
                pending.faceFrame = best_face_frame;
                pending.faceOcclusion = frame_occlusion(best_face_frame);
            }
        }

        if (!is_track_person_captured(t.id)) {
            if (pending.uploadFace) {
                pending.uploadPerson = true;
                pending.personFrame = pending.faceFrame;
            } else if (best_person_index != SIZE_MAX) {
                const auto& best_person_frame = t.frame_candidates[best_person_index];
                bool area_ok = best_person_frame.area_ratio >= person_area_threshold;
                bool occlusion_ok = best_person_frame.person_occlusion <=
                    std::max(g_capture_max_person_occlusion * 1.10f, 0.62f);
                if (area_ok && occlusion_ok) {
                    pending.uploadPerson = true;
                    pending.personFrame = best_person_frame;
                }
            }
        }

        if (!pending.uploadPerson && !pending.uploadFace) {
            // ── 无脸人体兜底上传 ──
            // 当没有合格人脸时，仍然上传最佳人体全身图，避免漏抓侧脸/低头/远距离人员。
            if (!is_track_person_captured(t.id) && best_person_index != SIZE_MAX) {
                const auto& fallback_person = t.frame_candidates[best_person_index];
                // 兜底上传使用更宽松的面积和遮挡门槛
                float fallback_area_threshold =
                    std::max(g_capture_min_area_ratio * 0.90f, g_capture_near_area_ratio * 0.65f);
                float fallback_occlusion_max =
                    std::max(g_capture_max_person_occlusion * 1.25f, 0.68f);
                bool fallback_area_ok = fallback_person.area_ratio >= fallback_area_threshold;
                bool fallback_occ_ok = fallback_person.person_occlusion <= fallback_occlusion_max;
                if (fallback_area_ok && fallback_occ_ok) {
                    pending.uploadPerson = true;
                    pending.personFrame = fallback_person;
                    log_info("Track %d fallback person-only upload: no usable face, area=%.4f occ=%.2f clarity=%.1f",
                             t.id,
                             fallback_person.area_ratio,
                             fallback_person.person_occlusion,
                             fallback_person.clarity);
                }
            }
        }

        if (!pending.uploadPerson && !pending.uploadFace) {
            float best_area_ratio = 0.0f;
            if (best_face_index != SIZE_MAX) {
                best_area_ratio = std::max(best_area_ratio, t.frame_candidates[best_face_index].area_ratio);
            }
            if (best_person_index != SIZE_MAX) {
                best_area_ratio = std::max(best_area_ratio, t.frame_candidates[best_person_index].area_ratio);
            }
            log_debug("Track %d skipped upload: target still too far or too occluded (best_area=%.4f)",
                      t.id,
                      best_area_ratio);
            return;
        }

        TrackTrajectoryDecision trajectory_decision;
        const char* trajectory_reason = nullptr;
        if (!should_upload_track_by_trajectory(t, &trajectory_decision, &trajectory_reason)) {
            log_info("---Track %d skipped upload on loss: reason=%s dir=%s score=%.3f peak_score=%.3f return_score=%.3f reversed=%d samples=%d bottom=%.3f->%.3f peak=%.3f area=%.4f->%.4f peak=%.4f max_area=%.4f",
                     t.id,
                     trajectory_reason ? trajectory_reason : "trajectory_reject",
                     track_trajectory_direction_to_string(trajectory_decision.direction),
                     trajectory_decision.score,
                     trajectory_decision.approach_peak_score,
                     trajectory_decision.return_score,
                     trajectory_decision.reversed_after_approach ? 1 : 0,
                     trajectory_decision.samples,
                     trajectory_decision.bottom_start,
                     trajectory_decision.bottom_end,
                     trajectory_decision.peak_bottom,
                     trajectory_decision.area_start,
                     trajectory_decision.area_end,
                     trajectory_decision.peak_area,
                     t.max_area_ratio);
            return;
        }

        log_info("Track %d accepted upload by full trajectory: reason=%s dir=%s score=%.3f peak_score=%.3f return_score=%.3f samples=%d bottom=%.3f->%.3f peak=%.3f area=%.4f->%.4f peak=%.4f max_area=%.4f",
                 t.id,
                 trajectory_reason ? trajectory_reason : "trajectory_accept",
                 track_trajectory_direction_to_string(trajectory_decision.direction),
                 trajectory_decision.score,
                 trajectory_decision.approach_peak_score,
                 trajectory_decision.return_score,
                 trajectory_decision.samples,
                 trajectory_decision.bottom_start,
                 trajectory_decision.bottom_end,
                 trajectory_decision.peak_bottom,
                 trajectory_decision.area_start,
                 trajectory_decision.area_end,
                 trajectory_decision.peak_area,
                 t.max_area_ratio);

        pendingUploads.push_back(std::move(pending));
        if (pendingUploads.back().uploadPerson) {
            captured_person_ids->insert(t.id);
        }
        if (pendingUploads.back().uploadFace) {
            captured_face_ids->insert(t.id);
        }
        remember_recent_capture(t);
    };

    // 棰勬祴鎵€鏈塼rack
    for (auto& t : tracks) predict_track_robust(t);

    int N = tracks.size();
    int M = dets.size();
    
    if (N == 0) {
        // 娌℃湁鐜版湁track鏃朵篃涓嶇珛鍗冲缓杞紝鍏堣繘鍏ending纭
        std::vector<cv::Mat> det_hists(M);
        for (int j = 0; j < M; j++) {
            det_hists[j] = calc_hist(dets[j].roi);
        }
        for (int j = 0; j < M; j++) {
            if (!dets[j].allow_new_track) {
                continue;
            }
            int assigned_id = update_pending_track(dets[j], det_hists[j]);
            if (assigned_id > 0) {
                log_debug("New person appeared: ID=%d", assigned_id);
            }
        }
        return make_track_snapshots(tracks);
    }
    
    if (M == 0) {
        auto it = std::remove_if(tracks.begin(), tracks.end(),
                    [&](const Track& t){
                        if(t.missed > MAX_MISSED){
                            cache_lost_track(t);
                            queue_upload_if_needed(t);
                            return true;
                        }
                        return false;
                    });
        tracks.erase(it, tracks.end());
        snapshot = make_track_snapshots(tracks);
        lock.unlock();
        for (const auto& upload : pendingUploads) {
            if (upload.uploadPerson && !upload.personFrame.person_roi.empty()) {
                upload_callback(upload.personFrame.person_roi,
                                upload.trackId,
                                upload.uploadFace ? "person" : "person_only");
            }
            if (upload.uploadFace && !upload.faceFrame.face_roi.empty()) {
                upload_callback(upload.faceFrame.face_roi, upload.trackId, "face");
            }
            const auto& log_frame = upload.uploadFace ? upload.faceFrame : upload.personFrame;
            log_info("Track %d upload queued: person=%d face=%d clarity=%.2f area=%.2f%% occ=%.2f motion=%.4f blur=%.2f score=%.2f",
                     upload.trackId,
                     upload.uploadPerson ? 1 : 0,
                     upload.uploadFace ? 1 : 0,
                     log_frame.clarity,
                     log_frame.area_ratio * 100.0f,
                     upload.uploadFace ? upload.faceOcclusion : log_frame.person_occlusion,
                     log_frame.motion_ratio,
                     log_frame.blur_severity,
                     log_frame.score);
        }
        return snapshot;
    }

    std::vector<cv::Mat> det_hists(M);
    for (int j = 0; j < M; j++) {
        det_hists[j] = calc_hist(dets[j].roi);
    }

    // Pre-compute EKF predicted bbox for each track.
    std::vector<cv::Rect2f> predicted_bboxes(N);
    for (int i = 0; i < N; i++) {
        float px = tracks[i].ekf.x[0] + tracks[i].ekf.x[4];
        float py = tracks[i].ekf.x[1] + tracks[i].ekf.x[5];
        float pw = std::max(10.0f, tracks[i].ekf.x[2] + tracks[i].ekf.x[6]);
        float ph = std::max(10.0f, tracks[i].ekf.x[3] + tracks[i].ekf.x[7]);
        predicted_bboxes[i] = clamp_bbox(cv::Rect2f(px, py, pw, ph));
    }

    std::vector<int> high_det_indices;
    std::vector<int> low_det_indices;
    high_det_indices.reserve(M);
    low_det_indices.reserve(M);
    for (int j = 0; j < M; ++j) {
        if (dets[j].allow_new_track) {
            high_det_indices.push_back(j);
        } else {
            low_det_indices.push_back(j);
        }
    }

    std::vector<bool> track_assigned(N, false);
    std::vector<bool> det_assigned(M, false);

    auto run_matching_stage = [&](const std::vector<int>& track_indices,
                                  const std::vector<int>& det_indices,
                                  float max_cost,
                                  bool low_confidence_stage) {
        if (track_indices.empty() || det_indices.empty()) {
            return;
        }

        std::vector<std::vector<float>> cost(track_indices.size(),
                                             std::vector<float>(det_indices.size(), 1.0f));
        for (size_t row = 0; row < track_indices.size(); ++row) {
            int i = track_indices[row];
            for (size_t col = 0; col < det_indices.size(); ++col) {
                int j = det_indices[col];
                cv::Rect2f det_rect(dets[j].x1, dets[j].y1,
                                    dets[j].x2 - dets[j].x1,
                                    dets[j].y2 - dets[j].y1);

                cv::Rect2f stable_bbox = stable_track_bbox(tracks[i]);
                float iou_score = std::max({iou(tracks[i].bbox, det_rect),
                                            iou(stable_bbox, det_rect),
                                            iou(predicted_bboxes[i], det_rect)});

                float hist_score = hist_distance(tracks[i].hist, det_hists[j]);

                float center_dist = std::min({center_distance_norm(tracks[i].bbox, det_rect),
                                              center_distance_norm(stable_bbox, det_rect),
                                              center_distance_norm(predicted_bboxes[i], det_rect)});

                float conf_weight = std::min(1.0f, dets[j].prop / 0.8f);
                float area_ratio = std::min(stable_bbox.area(), det_rect.area()) /
                                   std::max(stable_bbox.area(), det_rect.area());

                float center_cost = std::min(1.0f, center_dist / 0.28f);
                float area_penalty = 0.0f;
                if (area_ratio < 0.55f) {
                    float appearance_support = (1.0f - hist_score) * 0.55f + (1.0f - center_cost) * 0.45f;
                    float severity = (0.55f - area_ratio) / 0.55f;
                    float max_penalty = appearance_support > 0.65f ? 0.18f : 0.34f;
                    area_penalty = severity * max_penalty;
                }

                cost[row][col] = (1.0f - iou_score) * 0.48f +
                                 hist_score * 0.18f +
                                 center_cost * 0.26f +
                                 (1.0f - conf_weight) * 0.05f +
                                 area_penalty;

                if (low_confidence_stage) {
                    cost[row][col] += 0.04f;
                }
                if (iou_score < 0.02f && center_dist > 0.24f) {
                    cost[row][col] += 0.30f;
                }
                if (center_dist > 0.42f) {
                    cost[row][col] += 0.25f;
                }
                if (hist_score > 0.82f) {
                    cost[row][col] += 0.20f;
                }
                if (tracks[i].confirmed) {
                    float pred_dist = center_distance_norm(predicted_bboxes[i], det_rect);
                    if (pred_dist > 0.18f) {
                        cost[row][col] += 0.12f;
                    }
                }
            }
        }

        std::vector<std::pair<int, int>> assignments = hungarian_algorithm(cost, max_cost);
        for (const auto& assignment : assignments) {
            int track_idx = track_indices[assignment.first];
            int det_idx = det_indices[assignment.second];
            if (track_assigned[track_idx] || det_assigned[det_idx]) {
                continue;
            }

            cv::Rect2f det_rect(dets[det_idx].x1, dets[det_idx].y1,
                                dets[det_idx].x2 - dets[det_idx].x1,
                                dets[det_idx].y2 - dets[det_idx].y1);
            cv::Rect2f stable_bbox = stable_track_bbox(tracks[track_idx]);
            float match_iou = std::max(iou(tracks[track_idx].bbox, det_rect),
                                       iou(stable_bbox, det_rect));
            float match_center_dist = std::min(center_distance_norm(tracks[track_idx].bbox, det_rect),
                                               center_distance_norm(stable_bbox, det_rect));
            if (tracks[track_idx].confirmed &&
                match_iou < 0.08f &&
                match_center_dist > (low_confidence_stage ? 0.24f : 0.20f)) {
                continue;
            }

            track_assigned[track_idx] = true;
            det_assigned[det_idx] = true;
            correct_track_robust(tracks[track_idx], dets[det_idx]);
        }
    };

    std::vector<int> all_track_indices;
    all_track_indices.reserve(N);
    for (int i = 0; i < N; ++i) {
        all_track_indices.push_back(i);
    }
    run_matching_stage(all_track_indices, high_det_indices, 0.70f, false);

    std::vector<int> unmatched_track_indices;
    unmatched_track_indices.reserve(N);
    for (int i = 0; i < N; ++i) {
        if (!track_assigned[i]) {
            unmatched_track_indices.push_back(i);
        }
    }
    run_matching_stage(unmatched_track_indices, low_det_indices, 0.72f, true);

    // 鍒涘缓鏂皌racks
    for (int j : high_det_indices) {
        if(!det_assigned[j] && dets[j].allow_new_track){
            if (should_suppress_new_track(dets[j])) {
                continue;
            }

            int assigned_id = update_pending_track(dets[j], det_hists[j]);
            if (assigned_id > 0) {
                log_debug("New person appeared: ID=%d", assigned_id);
            }
        }
    }
    
    // 鍒犻櫎闀挎湡涓㈠け鐨則racks锛屼絾鍦ㄥ垹闄ゅ墠鍏堝鐞嗕笂浼?
    auto it = std::remove_if(tracks.begin(), tracks.end(),
                [&](const Track& t){
                    if(t.missed > MAX_MISSED){
                        cache_lost_track(t);
                        queue_upload_if_needed(t);
                        return true;
                    }
                    return false;
                });
    tracks.erase(it, tracks.end());
    snapshot = make_track_snapshots(tracks);
    lock.unlock();
    for (const auto& upload : pendingUploads) {
        if (upload.uploadPerson && !upload.personFrame.person_roi.empty()) {
            upload_callback(upload.personFrame.person_roi,
                            upload.trackId,
                            upload.uploadFace ? "person" : "person_only");
        }
        if (upload.uploadFace && !upload.faceFrame.face_roi.empty()) {
            upload_callback(upload.faceFrame.face_roi, upload.trackId, "face");
        }
        const auto& log_frame = upload.uploadFace ? upload.faceFrame : upload.personFrame;
        log_info("Track %d upload queued: person=%d face=%d clarity=%.2f area=%.2f%% occ=%.2f motion=%.4f blur=%.2f score=%.2f",
                 upload.trackId,
                 upload.uploadPerson ? 1 : 0,
                 upload.uploadFace ? 1 : 0,
                 log_frame.clarity,
                 log_frame.area_ratio * 100.0f,
                 upload.uploadFace ? upload.faceOcclusion : log_frame.person_occlusion,
                 log_frame.motion_ratio,
                 log_frame.blur_severity,
                 log_frame.score);
    }
    return snapshot;
}

std::vector<TrackSnapshot> get_expiring_tracks() {
    std::lock_guard<std::mutex> lock(tracks_mutex);
    std::vector<TrackSnapshot> expiring_tracks;
    
    // 鎵惧埌鍗冲皢琚垹闄ょ殑tracks
    for (const auto& t : tracks) {
        if (t.missed > MAX_MISSED) {
            expiring_tracks.push_back(make_track_snapshot(t));
        }
    }
    
    return expiring_tracks;
}

std::vector<TrackSnapshot> sort_predict_only() {
    std::lock_guard<std::mutex> lock(tracks_mutex);
    for (auto& t : tracks) {
        if (t.missed <= MAX_MISSED) {
            predict_track_robust(t, false);
            // Smoothly advance smoothed_bbox toward EKF prediction.
            float alpha_c = TRACK_SMOOTH_CENTER_ALPHA * 0.5f;
            float alpha_s = TRACK_SMOOTH_SIZE_ALPHA * 0.5f;
            if (is_valid_bbox(t.smoothed_bbox)) {
                t.smoothed_bbox.x += alpha_c * (t.bbox.x - t.smoothed_bbox.x);
                t.smoothed_bbox.y += alpha_c * (t.bbox.y - t.smoothed_bbox.y);
                t.smoothed_bbox.width += alpha_s * (t.bbox.width - t.smoothed_bbox.width);
                t.smoothed_bbox.height += alpha_s * (t.bbox.height - t.smoothed_bbox.height);
            } else {
                t.smoothed_bbox = t.bbox;
            }
        }
    }
    return make_track_snapshots(tracks);
}

void add_frame_candidate(int track_id, const Track::FrameData& frame_data) {
    std::lock_guard<std::mutex> lock(tracks_mutex);
    for (auto& t : tracks) {
        if (t.id == track_id) {
            if (t.frame_candidates.size() < g_max_frame_candidates) {
                t.frame_candidates.push_back(frame_data);
                t.best_clarity = std::max(t.best_clarity, frame_data.clarity);
                log_debug("Track %d candidate stored: score=%.2f count=%zu face=%d area=%.4f",
                          track_id,
                          frame_data.score,
                          t.frame_candidates.size(),
                          is_usable_face_frame(frame_data) ? 1 : 0,
                          frame_data.area_ratio);
            } else {
                auto peak_clarity_it = std::max_element(t.frame_candidates.begin(), t.frame_candidates.end(),
                    [](const Track::FrameData& a, const Track::FrameData& b) {
                        return a.clarity < b.clarity;
                    });
                auto replace_it = t.frame_candidates.end();
                for (auto it = t.frame_candidates.begin(); it != t.frame_candidates.end(); ++it) {
                    bool protect_peak = (it == peak_clarity_it) &&
                                        peak_clarity_it->strong_candidate &&
                                        peak_clarity_it->face_pose_level >= 1 &&
                                        frame_data.clarity <= peak_clarity_it->clarity * 1.02;
                    if (protect_peak) {
                        continue;
                    }
                    if (replace_it == t.frame_candidates.end() ||
                        overall_capture_priority(*it) < overall_capture_priority(*replace_it)) {
                        replace_it = it;
                    }
                }

                if (replace_it == t.frame_candidates.end()) {
                    replace_it = peak_clarity_it;
                }

                bool replace = false;
                if (is_usable_face_frame(frame_data) || is_usable_face_frame(*replace_it)) {
                    replace = better_face_capture(frame_data, *replace_it) ||
                              overall_capture_priority(frame_data) > overall_capture_priority(*replace_it) + 8.0;
                } else {
                    replace = better_person_capture(frame_data, *replace_it) ||
                              person_capture_priority(frame_data) > person_capture_priority(*replace_it) + 10.0;
                }

                if (replace) {
                    *replace_it = frame_data;
                    t.best_clarity = std::max(t.best_clarity, frame_data.clarity);
                    log_debug("Track %d candidate replaced: score=%.2f face=%d area=%.4f",
                              track_id,
                              frame_data.score,
                              is_usable_face_frame(frame_data) ? 1 : 0,
                              frame_data.area_ratio);
                }
            }
            break;
        }
    }
}
