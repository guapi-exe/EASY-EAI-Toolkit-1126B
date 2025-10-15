#include "sort_tracker.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cstdio>
#include "tinyekf.h"
extern "C" {
#include "log.h"
}

static std::vector<Track> tracks;
static int next_id = 1;
static const int MAX_MISSED = 30;
static const int IMAGE_WIDTH = 1920;   // 图像宽度
static const int IMAGE_HEIGHT = 1080;  // 图像高度

void sort_init() { 
    tracks.clear(); 
    next_id = 1; 
}

//-----------------工具函数-----------------

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

//-----------------匈牙利算法-----------------

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

    // 边界检查，防止bbox超出图像边界
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

static Track create_track(const Detection& det, int id) {
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
    t.has_captured = false;
    return t;
}

//-----------------主更新函数-----------------

std::vector<Track> sort_update(const std::vector<Detection>& dets) {
    // 预测所有track
    for (auto& t : tracks) predict_track(t);

    int N = tracks.size();
    int M = dets.size();
    
    if (N == 0) {
        // 没有现有track，直接创建新的
        for (int j = 0; j < M; j++) {
            tracks.push_back(create_track(dets[j], next_id++));
            log_debug("New person appeared: ID=%d", next_id-1);
        }
        return tracks;
    }
    
    if (M == 0) {
        // 没有检测，只更新丢失计数
        tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
                    [](const Track& t){
                        if(t.missed>MAX_MISSED){
                            log_debug("Person disappeared: ID=%d", t.id);
                            return true;
                        }
                        return false;
                    }), tracks.end());
        return tracks;
    }

    // 计算代价矩阵
    std::vector<std::vector<float>> cost(N, std::vector<float>(M, 1.0f));

    for (int i=0; i<N; i++) {
        for (int j=0; j<M; j++) {
            cv::Rect2f det_rect(dets[j].x1, dets[j].y1, 
                               dets[j].x2-dets[j].x1, dets[j].y2-dets[j].y1);
            
            // IoU相似度 (0-1, 越大越好)
            float iou_score = iou(tracks[i].bbox, det_rect);
            
            // 颜色直方图距离 (0-1, 越小越好) 
            float hist_score = hist_distance(tracks[i].hist, calc_hist(dets[j].roi));
            
            // 置信度权重
            float conf_weight = std::min(1.0f, dets[j].prop / 0.8f);
            
            // 尺寸一致性检查
            float area_ratio = std::min(tracks[i].bbox.area(), det_rect.area()) / 
                              std::max(tracks[i].bbox.area(), det_rect.area());
            
            // 综合代价：IoU权重0.6, 直方图权重0.3, 置信度权重0.1
            cost[i][j] = (1.0f - iou_score) * 0.6f + 
                        hist_score * 0.3f + 
                        (1.0f - conf_weight) * 0.1f;
            
            // 如果尺寸差异过大，增加代价
            if (area_ratio < 0.3f) {
                cost[i][j] += 0.5f;
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
            tracks.push_back(create_track(dets[j], next_id++));
            log_debug("New person appeared: ID=%d", next_id-1);
        }
    }
    
    // 删除长期丢失的tracks
    tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
                [](const Track& t){
                    if(t.missed>MAX_MISSED){
                        log_debug("Person disappeared: ID=%d", t.id);
                        return true;
                    }
                    return false;
                }), tracks.end());

    return tracks;
}