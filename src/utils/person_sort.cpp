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
static constexpr size_t BBOX_HISTORY_LIMIT = 20;
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

static Track create_track(const Detection& det, int id, bool already_captured = false);

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
                                  float maxPersonOcclusion) {
    std::lock_guard<std::mutex> lock(tracks_mutex);
    g_capture_min_area_ratio = std::max(0.001f, minAreaRatio);
    g_capture_near_area_ratio = std::max(g_capture_min_area_ratio, nearAreaRatio);
    g_capture_max_person_occlusion = std::max(0.05f, maxPersonOcclusion);
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
    float near_bonus = std::min(1.8f, near_ratio) * 180.0f;
    float far_penalty = near_ratio < 1.0f ? (1.0f - near_ratio) * 240.0f : 0.0f;
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
    float near_bonus = std::min(1.8f, near_ratio) * 220.0f;
    float far_penalty = near_ratio < 1.0f ? (1.0f - near_ratio) * 300.0f : 0.0f;
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

static void predict_track_robust(Track& t) {
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
    Q[0 * EKF_N + 0] = 1.0f;
    Q[1 * EKF_N + 1] = 1.0f;
    Q[2 * EKF_N + 2] = 0.5f;
    Q[3 * EKF_N + 3] = 0.5f;
    Q[4 * EKF_N + 4] = 0.1f;
    Q[5 * EKF_N + 5] = 0.1f;
    Q[6 * EKF_N + 6] = 0.05f;
    Q[7 * EKF_N + 7] = 0.05f;

    ekf_predict(&t.ekf, t.ekf.x, F, Q);
    t.bbox = clamp_bbox(cv::Rect2f(t.ekf.x[0],
                                   t.ekf.x[1],
                                   std::max(10.0f, t.ekf.x[2]),
                                   std::max(10.0f, t.ekf.x[3])));
    t.age++;
    t.missed++;
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
    t.best_area = 0.0f;
    t.best_clarity = 0.0;
    t.has_captured = already_captured;
    return t;
}

//-----------------涓绘洿鏂板嚱鏁?----------------

std::vector<Track> sort_update(const std::vector<Detection>& dets) {
    struct PendingUpload {
        int trackId;
        bool uploadPerson{false};
        bool uploadFace{false};
        Track::FrameData personFrame;
        Track::FrameData faceFrame;
        float faceOcclusion{0.0f};
    };

    std::vector<PendingUpload> pendingUploads;
    std::vector<Track> snapshot;

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
            int assigned_id = update_pending_track(dets[j], det_hists[j]);
            if (assigned_id > 0) {
                log_debug("New person appeared: ID=%d", assigned_id);
            }
        }
        return tracks;
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
        snapshot = tracks;
        lock.unlock();
        for (const auto& upload : pendingUploads) {
            if (upload.uploadPerson && !upload.personFrame.person_roi.empty()) {
                upload_callback(upload.personFrame.person_roi, upload.trackId, "person");
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

    // Cost matrix construction — position-dominant weights to prevent ID swaps.
    std::vector<std::vector<float>> cost(N, std::vector<float>(M, 1.0f));

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

    for (int i=0; i<N; i++) {
        for (int j=0; j<M; j++) {
            cv::Rect2f det_rect(dets[j].x1, dets[j].y1,
                               dets[j].x2-dets[j].x1, dets[j].y2-dets[j].y1);

            cv::Rect2f stable_bbox = stable_track_bbox(tracks[i]);
            // IoU: take best of raw bbox, smoothed bbox, and EKF predicted bbox.
            float iou_score = std::max({iou(tracks[i].bbox, det_rect),
                                        iou(stable_bbox, det_rect),
                                        iou(predicted_bboxes[i], det_rect)});

            float hist_score = hist_distance(tracks[i].hist, det_hists[j]);

            // Center distance: take minimum of all three references.
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

            // Weights: position-dominant to prevent ID swaps between similar-looking people.
            // IoU 0.48 + hist 0.18 + center 0.26 + conf 0.05 = 0.97 + area_penalty
            cost[i][j] = (1.0f - iou_score) * 0.48f +
                        hist_score * 0.18f +
                        center_cost * 0.26f +
                        (1.0f - conf_weight) * 0.05f +
                        area_penalty;

            if (iou_score < 0.02f && center_dist > 0.24f) {
                cost[i][j] += 0.30f;
            }

            if (center_dist > 0.42f) {
                cost[i][j] += 0.25f;
            }

            if (hist_score > 0.82f) {
                cost[i][j] += 0.20f;
            }

            // Velocity consistency penalty: if EKF predicts the track should be
            // far from this detection, penalize the match (prevents ID swaps).
            if (tracks[i].confirmed) {
                float pred_dist = center_distance_norm(predicted_bboxes[i], det_rect);
                if (pred_dist > 0.15f) {
                    cost[i][j] += 0.15f;
                }
            }
        }
    }

    // max_cost tightened to reject weak matches that cause ID swaps.
    std::vector<std::pair<int,int>> assignments = hungarian_algorithm(cost, 0.65f);
    
    std::vector<bool> track_assigned(N, false);
    std::vector<bool> det_assigned(M, false);

    // Apply assignments with post-match validation to prevent ID swaps.
    for (const auto& assignment : assignments) {
        int track_idx = assignment.first;
        int det_idx = assignment.second;

        // Post-match IoU gate: reject matches where the detection is spatially
        // far from the track, even if the Hungarian algorithm chose it as "optimal".
        // This prevents cross-assignments between similar-looking people.
        if (tracks[track_idx].confirmed) {
            cv::Rect2f det_rect(dets[det_idx].x1, dets[det_idx].y1,
                               dets[det_idx].x2 - dets[det_idx].x1,
                               dets[det_idx].y2 - dets[det_idx].y1);
            cv::Rect2f stable_bbox = stable_track_bbox(tracks[track_idx]);
            float match_iou = std::max(iou(tracks[track_idx].bbox, det_rect),
                                       iou(stable_bbox, det_rect));
            float match_center_dist = std::min(center_distance_norm(tracks[track_idx].bbox, det_rect),
                                               center_distance_norm(stable_bbox, det_rect));
            if (match_iou < 0.08f && match_center_dist > 0.20f) {
                // Reject: this match is spatially implausible for a confirmed track.
                continue;
            }
        }

        track_assigned[track_idx] = true;
        det_assigned[det_idx] = true;
        correct_track_robust(tracks[track_idx], dets[det_idx]);
    }

    // 鍒涘缓鏂皌racks
    for (int j=0; j<M; j++) {
        if(!det_assigned[j]){
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
    snapshot = tracks;
    lock.unlock();
    for (const auto& upload : pendingUploads) {
        if (upload.uploadPerson && !upload.personFrame.person_roi.empty()) {
            upload_callback(upload.personFrame.person_roi, upload.trackId, "person");
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

std::vector<Track> get_expiring_tracks() {
    std::lock_guard<std::mutex> lock(tracks_mutex);
    std::vector<Track> expiring_tracks;
    
    // 鎵惧埌鍗冲皢琚垹闄ょ殑tracks
    for (const auto& t : tracks) {
        if (t.missed > MAX_MISSED) {
            expiring_tracks.push_back(t);
        }
    }
    
    return expiring_tracks;
}

std::vector<Track> sort_predict_only() {
    std::lock_guard<std::mutex> lock(tracks_mutex);
    for (auto& t : tracks) {
        if (t.missed <= MAX_MISSED) {
            predict_track_robust(t);
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
    return tracks;
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
