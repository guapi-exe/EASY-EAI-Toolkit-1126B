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
static constexpr int LOST_TRACK_TTL = MAX_MISSED + 15;
static constexpr int RECENT_CAPTURE_TTL = MAX_MISSED + 90;
static constexpr int PENDING_TRACK_TTL = 4;
static constexpr int PENDING_TRACK_HITS_REQUIRED = 2;

static Track create_track(const Detection& det, int id, bool already_captured = false);

void sort_init() { 
    std::unique_lock<std::mutex> lock(tracks_mutex);
    tracks.clear(); 
    lost_tracks.clear();
    recent_captures.clear();
    pending_tracks.clear();
    next_id = 1; 
}

// 添加上传回调函数指针
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

static bool is_track_already_captured(int track_id) {
    return (captured_person_ids && captured_person_ids->find(track_id) != captured_person_ids->end()) ||
           (captured_face_ids && captured_face_ids->find(track_id) != captured_face_ids->end());
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
    lt.bbox = t.bbox;
    lt.hist = t.hist.clone();
    lt.ttl = LOST_TRACK_TTL;
    lost_tracks.push_back(std::move(lt));
}

static void remember_recent_capture(const Track& t) {
    if (!is_track_already_captured(t.id) || t.hist.empty()) {
        return;
    }

    recent_captures.erase(
        std::remove_if(recent_captures.begin(), recent_captures.end(), [&](const RecentCapture& rc) {
            return rc.id == t.id;
        }),
        recent_captures.end());

    RecentCapture rc;
    rc.id = t.id;
    rc.bbox = t.bbox;
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
        float center_score = 1.0f - std::min(1.0f, center_distance_norm(lt.bbox, det_rect) / 0.35f);

        if (iou_score < 0.05f || hist_score > 0.6f || center_score < 0.0f) {
            continue;
        }

        float score = iou_score * 0.55f + (1.0f - hist_score) * 0.30f + center_score * 0.15f;
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    if (best_idx != -1 && best_score >= 0.45f) {
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
        float center_score = 1.0f - std::min(1.0f, center_dist / 0.18f);

        if (hist_score > 0.34f) {
            continue;
        }
        if (iou_score < 0.06f && center_dist > 0.12f) {
            continue;
        }

        float score = (1.0f - hist_score) * 0.55f + center_score * 0.30f + iou_score * 0.15f;
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    if (best_idx != -1 && best_score >= 0.67f) {
        int reused_id = recent_captures[best_idx].id;
        recent_captures.erase(recent_captures.begin() + best_idx);
        log_debug("Reuse recent captured id: %d (score=%.3f)", reused_id, best_score);
        return reused_id;
    }

    return -1;
}

static size_t select_best_frame_index(const std::vector<Track::FrameData>& frames) {
    size_t best_index = SIZE_MAX;
    int best_pose_level = -1;
    bool best_strong_candidate = false;
    float best_yaw = 1e6f;
    double best_score = -1e12;
    float best_occlusion = 1e6f;
    float best_motion = 1e6f;
    float best_blur = 1e6f;
    double best_clarity = -1e12;

    auto occlusion_of = [](const Track::FrameData& frame) {
        return frame.person_occlusion * 0.7f + frame.face_edge_occlusion * 0.3f;
    };

    for (size_t i = 0; i < frames.size(); ++i) {
        const auto& frame = frames[i];
        if (!frame.has_face || frame.face_pose_level < 1) {
            continue;
        }

        float frame_occlusion = occlusion_of(frame);
        bool clearly_lower_blur = frame.blur_severity < best_blur - 0.05f;
        bool similar_blur = std::fabs(frame.blur_severity - best_blur) <= 0.05f;
        bool clearly_lower_motion = similar_blur && frame.motion_ratio < best_motion - 0.003f;
        bool similar_motion = similar_blur && std::fabs(frame.motion_ratio - best_motion) <= 0.003f;

        bool better = false;
        if (best_index == SIZE_MAX || frame.face_pose_level > best_pose_level) {
            better = true;
        } else if (frame.face_pose_level == best_pose_level) {
            if (frame.strong_candidate != best_strong_candidate) {
                better = frame.strong_candidate && !best_strong_candidate;
            } else if (clearly_lower_blur) {
                better = true;
            } else if (similar_blur && clearly_lower_motion) {
                better = true;
            } else if (similar_motion && frame_occlusion < best_occlusion - 0.02f) {
                better = true;
            } else if (similar_motion &&
                       std::fabs(frame_occlusion - best_occlusion) <= 0.02f &&
                       frame.clarity > best_clarity + 1e-6f) {
                better = true;
            } else if (similar_motion &&
                       std::fabs(frame_occlusion - best_occlusion) <= 0.02f &&
                       std::fabs(frame.clarity - best_clarity) <= 1e-6f &&
                       frame.yaw_abs < best_yaw - 1e-6f) {
                better = true;
            } else if (similar_motion &&
                       std::fabs(frame_occlusion - best_occlusion) <= 0.02f &&
                       std::fabs(frame.clarity - best_clarity) <= 1e-6f &&
                       std::fabs(frame.yaw_abs - best_yaw) <= 1e-6f &&
                       frame.score > best_score) {
                better = true;
            }
        }

        if (better) {
            best_pose_level = frame.face_pose_level;
            best_strong_candidate = frame.strong_candidate;
            best_blur = frame.blur_severity;
            best_occlusion = frame_occlusion;
            best_motion = frame.motion_ratio;
            best_clarity = frame.clarity;
            best_yaw = frame.yaw_abs;
            best_score = frame.score;
            best_index = i;
        }
    }

    return best_index;
}

static bool should_suppress_new_track(const Detection& det) {
    cv::Rect2f det_rect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);

    for (const auto& t : tracks) {
        float iou_score = iou(t.bbox, det_rect);
        float center_dist = center_distance_norm(t.bbox, det_rect);
        if (iou_score > 0.45f || center_dist < 0.03f) {
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
        float score = iou_score * 0.65f + (1.0f - hist_score) * 0.35f;
        if (iou_score > 0.20f && hist_score < 0.55f && score > best_score) {
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
    pt.bbox = det_rect;
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
        bool already_captured = is_track_already_captured(assigned_id);

        tracks.push_back(create_track(det, assigned_id, already_captured));
        pending_tracks.erase(pending_tracks.begin() + best_idx);
        return assigned_id;
    }

    return -1;
}


static std::vector<std::pair<int,int>> hungarian_algorithm(const std::vector<std::vector<float>>& cost_matrix, float max_cost) {
    int n = cost_matrix.size();
    int m = cost_matrix[0].size();
    
    std::vector<std::pair<int,int>> assignments;
    
    // 简化的匈牙利算法实现
    std::vector<bool> row_assigned(n, false);
    std::vector<bool> col_assigned(m, false);
    
    // 首先进行贪婪匹配，但确保全局最优
    for(int iter = 0; iter < std::min(n, m); iter++) {
        float min_cost = 1e6;
        int best_i = -1, best_j = -1;
        
        for(int i = 0; i < n; i++) {
            if(row_assigned[i]) continue;
            for(int j = 0; j < m; j++) {
                if(col_assigned[j]) continue;
                if(cost_matrix[i][j] < min_cost) {
                    min_cost = cost_matrix[i][j];
                    best_i = i;
                    best_j = j;
                }
            }
        }
        
        if(best_i != -1 && best_j != -1 && min_cost < max_cost) {
            assignments.push_back({best_i, best_j});
            row_assigned[best_i] = true;
            col_assigned[best_j] = true;
        } else {
            break;
        }
    }
    
    return assignments;
}

//-----------------Track EKF操作-----------------

static void predict_track(Track& t) {
    // 改进的8状态运动模型: [x, y, w, h, vx, vy, vw, vh]
    _float_t dt = 1.0f;  // 时间步长
    _float_t F[EKF_N*EKF_N] = {
        1,0,0,0,dt, 0, 0, 0,  // x = x + vx*dt
        0,1,0,0, 0,dt, 0, 0,  // y = y + vy*dt  
        0,0,1,0, 0, 0,dt, 0,  // w = w + vw*dt
        0,0,0,1, 0, 0, 0,dt,  // h = h + vh*dt
        0,0,0,0, 1, 0, 0, 0,  // vx = vx (常速度模型)
        0,0,0,0, 0, 1, 0, 0,  // vy = vy
        0,0,0,0, 0, 0, 1, 0,  // vw = vw  
        0,0,0,0, 0, 0, 0, 1   // vh = vh
    };
    
    // 改进的过程噪声矩阵 - 根据运动不确定性调整
    _float_t Q[EKF_N*EKF_N] = {0};
    // 位置噪声
    Q[0*EKF_N+0] = 1.0f;   // x位置噪声
    Q[1*EKF_N+1] = 1.0f;   // y位置噪声
    Q[2*EKF_N+2] = 0.5f;   // width噪声(较小)
    Q[3*EKF_N+3] = 0.5f;   // height噪声(较小)
    // 速度噪声
    Q[4*EKF_N+4] = 0.1f;   // x速度噪声
    Q[5*EKF_N+5] = 0.1f;   // y速度噪声  
    Q[6*EKF_N+6] = 0.05f;  // width变化速度噪声
    Q[7*EKF_N+7] = 0.05f;  // height变化速度噪声
    
    ekf_predict(&t.ekf, t.ekf.x, F, Q);

    t.bbox.x = t.ekf.x[0];
    t.bbox.y = t.ekf.x[1];
    t.bbox.width  = std::max(10.0f, t.ekf.x[2]);   // 防止宽度过小
    t.bbox.height = std::max(10.0f, t.ekf.x[3]);   // 防止高度过小

    // 边界检查，防止bbox超出图像边界（现在使用720p坐标系）
    t.bbox.x = std::max(0.0f, std::min((float)(IMAGE_WIDTH - t.bbox.width), t.bbox.x));
    t.bbox.y = std::max(0.0f, std::min((float)(IMAGE_HEIGHT - t.bbox.height), t.bbox.y));
    
    // 确保bbox在图像范围内
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
    _float_t R[EKF_M*EKF_M] = {0.01,0,0,0,
                                0,0.01,0,0,
                                0,0,0.01,0,
                                0,0,0,0.01};
    ekf_update(&t.ekf, z, z, H, R);

    t.bbox = cv::Rect2f(det.x1, det.y1, det.x2-det.x1, det.y2-det.y1);
    t.hist = calc_hist(det.roi);
    t.prop = det.prop;  // 更新置信度
    t.missed = 0;
    t.hits++;
    t.active = true;
    
    // 经过3次成功匹配后确认track
    if (t.hits >= 3) {
        t.confirmed = true;
    }

    // 记录检测框面积历史
    float area = t.bbox.width * t.bbox.height;
    t.bbox_history.push_back(area);
    // 只保留最近20帧
    if (t.bbox_history.size() > 20) t.bbox_history.erase(t.bbox_history.begin());
}

//-----------------新建Track-----------------

static Track create_track(const Detection& det, int id, bool already_captured) {
    Track t;
    t.id = id;

    _float_t Pdiag[EKF_N] = {1,1,1,1,10,10,10,10};  // 位置方差较小，速度方差较大
    ekf_initialize(&t.ekf, Pdiag);

    _float_t state[EKF_N] = {det.x1, det.y1, det.x2-det.x1, det.y2-det.y1, 0, 0, 0, 0};
    memcpy(t.ekf.x, state, sizeof(state));

    t.bbox = cv::Rect2f(det.x1, det.y1, det.x2-det.x1, det.y2-det.y1);
    t.hist = calc_hist(det.roi);
    t.prop = det.prop;
    t.age = 1;
    t.missed = 0;
    t.hits = 1;  // 初始命中次数
    t.active = true;
    t.confirmed = false;  // 需要几帧确认
    t.is_approaching = false;
    t.best_area = 0.0f;
    t.best_clarity = 0.0;
    t.has_captured = already_captured;
    return t;
}

//-----------------主更新函数-----------------

std::vector<Track> sort_update(const std::vector<Detection>& dets) {
    struct PendingUpload {
        int trackId;
        Track::FrameData bestFrame;
        float occlusion;
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
            if (is_track_already_captured(t.id)) {
                remember_recent_capture(t);
            }
            return;
        }

        if (is_track_already_captured(t.id)) {
            remember_recent_capture(t);
            log_debug("Track %d already captured, refresh recent cache only", t.id);
            return;
        }

        size_t best_index = select_best_frame_index(t.frame_candidates);
        if (best_index == SIZE_MAX) {
            log_debug("Track %d skipped upload: no frontal-quality candidate", t.id);
            return;
        }

        const auto& best_frame = t.frame_candidates[best_index];
        float occ = best_frame.person_occlusion * 0.7f + best_frame.face_edge_occlusion * 0.3f;
        pendingUploads.push_back({t.id, best_frame, occ});
        captured_person_ids->insert(t.id);
        captured_face_ids->insert(t.id);
        remember_recent_capture(t);
    };

    // 预测所有track
    for (auto& t : tracks) predict_track(t);

    int N = tracks.size();
    int M = dets.size();
    
    if (N == 0) {
        // 没有现有track时也不立即建轨，先进入pending确认
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
            upload_callback(upload.bestFrame.person_roi, upload.trackId, "person");
            upload_callback(upload.bestFrame.face_roi, upload.trackId, "face");
            log_info("Track %d 上传最佳帧 (清晰度: %.2f, 面积占比: %.2f%%, 遮挡: %.2f, 运动: %.4f, 模糊度: %.2f, 综合评分: %.2f)",
                     upload.trackId,
                     upload.bestFrame.clarity,
                     upload.bestFrame.area_ratio * 100,
                     upload.occlusion,
                     upload.bestFrame.motion_ratio,
                     upload.bestFrame.blur_severity,
                     upload.bestFrame.score);
        }
        return snapshot;
    }

    // 计算代价矩阵
    std::vector<std::vector<float>> cost(N, std::vector<float>(M, 1.0f));

    std::vector<cv::Mat> det_hists(M);
    for (int j = 0; j < M; j++) {
        det_hists[j] = calc_hist(dets[j].roi);
    }

    for (int i=0; i<N; i++) {
        for (int j=0; j<M; j++) {
            cv::Rect2f det_rect(dets[j].x1, dets[j].y1, 
                               dets[j].x2-dets[j].x1, dets[j].y2-dets[j].y1);
            
            // IoU相似度 (0-1, 越大越好)
            float iou_score = iou(tracks[i].bbox, det_rect);
            
            // 颜色直方图距离 (0-1, 越小越好) 
            float hist_score = hist_distance(tracks[i].hist, det_hists[j]);

            // 中心点距离归一化（低帧率下比IoU更稳）
            float center_dist = center_distance_norm(tracks[i].bbox, det_rect);
            
            // 置信度权重
            float conf_weight = std::min(1.0f, dets[j].prop / 0.8f);
            
            // 尺寸一致性检查
            float area_ratio = std::min(tracks[i].bbox.area(), det_rect.area()) / 
                              std::max(tracks[i].bbox.area(), det_rect.area());
            
            // 综合代价：IoU权重0.5, 直方图权重0.25, 中心距离权重0.2, 置信度权重0.05
            cost[i][j] = (1.0f - iou_score) * 0.5f + 
                        hist_score * 0.25f +
                        std::min(1.0f, center_dist / 0.25f) * 0.2f +
                        (1.0f - conf_weight) * 0.05f;
            
            // 如果尺寸差异过大，增加代价
            if (area_ratio < 0.3f) {
                cost[i][j] += 0.5f;
            }

            if (center_dist > 0.35f) {
                cost[i][j] += 0.4f;
            }
        }
    }

    // 使用匈牙利算法进行最优匹配
    std::vector<std::pair<int,int>> assignments = hungarian_algorithm(cost, 0.7f);
    
    std::vector<bool> track_assigned(N, false);
    std::vector<bool> det_assigned(M, false);

    // 应用匹配结果
    for (const auto& assignment : assignments) {
        int track_idx = assignment.first;
        int det_idx = assignment.second;
        
        track_assigned[track_idx] = true;
        det_assigned[det_idx] = true;
        correct_track(tracks[track_idx], dets[det_idx]);
    }

    // 创建新tracks
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
    
    // 删除长期丢失的tracks，但在删除前先处理上传
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
        upload_callback(upload.bestFrame.person_roi, upload.trackId, "person");
        upload_callback(upload.bestFrame.face_roi, upload.trackId, "face");
        log_info("Track %d 上传最佳帧 (清晰度: %.2f, 面积占比: %.2f%%, 遮挡: %.2f, 运动: %.4f, 模糊度: %.2f, 综合评分: %.2f)",
                 upload.trackId,
                 upload.bestFrame.clarity,
                 upload.bestFrame.area_ratio * 100,
                 upload.occlusion,
                 upload.bestFrame.motion_ratio,
                 upload.bestFrame.blur_severity,
                 upload.bestFrame.score);
    }

    return snapshot;
}

std::vector<Track> get_expiring_tracks() {
    std::lock_guard<std::mutex> lock(tracks_mutex);
    std::vector<Track> expiring_tracks;
    
    // 找到即将被删除的tracks
    for (const auto& t : tracks) {
        if (t.missed > MAX_MISSED) {
            expiring_tracks.push_back(t);
        }
    }
    
    return expiring_tracks;
}

void add_frame_candidate(int track_id, const Track::FrameData& frame_data) {
    std::lock_guard<std::mutex> lock(tracks_mutex);
    for (auto& t : tracks) {
        if (t.id == track_id) {
            if (t.frame_candidates.size() < g_max_frame_candidates) {
                t.frame_candidates.push_back(frame_data);
                log_debug("Track %d 添加候选帧 (评分: %.2f, 候选帧总数: %zu)", 
                         track_id, frame_data.score, t.frame_candidates.size());
            } else {
                auto min_it = std::min_element(t.frame_candidates.begin(), t.frame_candidates.end(),
                    [](const Track::FrameData& a, const Track::FrameData& b) {
                        return a.score < b.score;
                    });
                
                if (frame_data.score > min_it->score) {
                    *min_it = frame_data;
                    log_debug("Track %d 替换低分帧 (新分数: %.2f)", track_id, frame_data.score);
                }
            }
            break;
        }
    }
}
