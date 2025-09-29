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

//-----------------Track EKF操作-----------------

static void predict_track(Track& t) {
    _float_t F[EKF_N*EKF_N] = {1,0,1,0,
                                0,1,0,1,
                                0,0,1,0,
                                0,0,0,1};
    _float_t Q[EKF_N*EKF_N] = {0};  // 可调小过程噪声
    ekf_predict(&t.ekf, t.ekf.x, F, Q);

    t.bbox.x = t.ekf.x[0];
    t.bbox.y = t.ekf.x[1];
    t.bbox.width  = t.ekf.x[2];
    t.bbox.height = t.ekf.x[3];

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
    t.missed = 0;
    t.active = true;
}

//-----------------新建Track-----------------

static Track create_track(const Detection& det, int id) {
    Track t;
    t.id = id;

    _float_t Pdiag[EKF_N] = {1,1,1,1};
    ekf_initialize(&t.ekf, Pdiag);

    _float_t state[EKF_N] = {det.x1, det.y1, det.x2-det.x1, det.y2-det.y1};
    memcpy(t.ekf.x, state, sizeof(state));

    t.bbox = cv::Rect2f(det.x1, det.y1, det.x2-det.x1, det.y2-det.y1);
    t.hist = calc_hist(det.roi);
    t.age = 1;
    t.missed = 0;
    t.active = true;
    return t;
}

//-----------------主更新函数-----------------

std::vector<Track> sort_update(const std::vector<Detection>& dets) {
    for (auto& t : tracks) predict_track(t);

    int N = tracks.size();
    int M = dets.size();
    std::vector<std::vector<float>> cost(N, std::vector<float>(M, 1.0f));

    for (int i=0; i<N; i++) {
        for (int j=0; j<M; j++) {
            float iou_score = iou(tracks[i].bbox, cv::Rect2f(dets[j].x1,dets[j].y1,
                                                              dets[j].x2-dets[j].x1,
                                                              dets[j].y2-dets[j].y1));
            float hist_score = hist_distance(tracks[i].hist, calc_hist(dets[j].roi));
            cost[i][j] = 1.0f - iou_score + hist_score;
        }
    }

    std::vector<int> det_assigned(M,-1);
    std::vector<bool> track_assigned(N,false);

    for (int j=0; j<M; j++) {
        float min_cost = 1e6;
        int best_i = -1;
        for (int i=0; i<N; i++) {
            if(track_assigned[i]) continue;
            if(cost[i][j] < 0.7f && cost[i][j] < min_cost){
                min_cost = cost[i][j];
                best_i = i;
            }
        }
        if(best_i != -1){
            track_assigned[best_i] = true;
            det_assigned[j] = best_i;
            correct_track(tracks[best_i], dets[j]);
        }
    }

    for (int j=0; j<M; j++) {
        if(det_assigned[j] == -1){
            tracks.push_back(create_track(dets[j], next_id++));
            log_debug("New person appeared: ID=%d\n", next_id-1);
        }
    }

    tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
                [](const Track& t){
                    if(t.missed>MAX_MISSED){
                        log_debug("Person disappeared: ID=%d\n", t.id);
                        return true;
                    }
                    return false;
                }), tracks.end());

    return tracks;
}