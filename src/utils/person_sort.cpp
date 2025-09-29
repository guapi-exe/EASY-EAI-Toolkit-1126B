#include "sort_tracker.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cstdio>
extern "C" {
#include "log.h"
}

static std::vector<Track> tracks;
static int next_id = 1;
static const int MAX_MISSED = 30;


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


std::vector<Track> sort_update(const std::vector<Detection>& dets) {
    for (auto& t : tracks) {
        cv::Mat prediction = t.kf.predict();
        t.bbox.x = prediction.at<float>(0);
        t.bbox.y = prediction.at<float>(1);
        t.bbox.width  = prediction.at<float>(2);
        t.bbox.height = prediction.at<float>(3);
        t.age++;
        t.missed++;
    }

    int N = tracks.size();
    int M = dets.size();
    std::vector<std::vector<float>> cost(N, std::vector<float>(M, 1.0f)); // 1 - IOU
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            float iou_score = iou(tracks[i].bbox, cv::Rect2f(dets[j].x1, dets[j].y1,
                                                              dets[j].x2 - dets[j].x1,
                                                              dets[j].y2 - dets[j].y1));
            float hist_score = hist_distance(tracks[i].hist, calc_hist(dets[j].roi));
            cost[i][j] = 1.0f - iou_score + hist_score; // 越小越匹配
        }
    }

    std::vector<int> det_assigned(M, -1);
    std::vector<bool> track_assigned(N, false);

    for (int j = 0; j < M; j++) {
        float min_cost = 1e6;
        int best_i = -1;
        for (int i = 0; i < N; i++) {
            if (track_assigned[i]) continue;
            if (cost[i][j] < 0.7f && cost[i][j] < min_cost) { // 匹配阈值
                min_cost = cost[i][j];
                best_i = i;
            }
        }
        if (best_i != -1) {
            track_assigned[best_i] = true;
            det_assigned[j] = best_i;

            // 修正 Kalman
            cv::Mat measurement(4,1,CV_32F);
            measurement.at<float>(0) = dets[j].x1;
            measurement.at<float>(1) = dets[j].y1;
            measurement.at<float>(2) = dets[j].x2 - dets[j].x1;
            measurement.at<float>(3) = dets[j].y2 - dets[j].y1;
            tracks[best_i].kf.correct(measurement);

            tracks[best_i].bbox = cv::Rect2f(dets[j].x1, dets[j].y1,
                                             dets[j].x2 - dets[j].x1,
                                             dets[j].y2 - dets[j].y1);
            tracks[best_i].hist = calc_hist(dets[j].roi);
            tracks[best_i].missed = 0;
            tracks[best_i].active = true;
        }
    }

    for (int j = 0; j < M; j++) {
        if (det_assigned[j] == -1) {
            Track t;
            t.id = next_id++;
            t.kf = cv::KalmanFilter(4,4,0);
            t.kf.transitionMatrix = (cv::Mat_<float>(4,4) << 
                                      1,0,1,0,
                                      0,1,0,1,
                                      0,0,1,0,
                                      0,0,0,1);
            cv::setIdentity(t.kf.measurementMatrix);
            cv::setIdentity(t.kf.processNoiseCov, cv::Scalar::all(1e-2));
            cv::setIdentity(t.kf.measurementNoiseCov, cv::Scalar::all(1e-1));
            cv::setIdentity(t.kf.errorCovPost, cv::Scalar::all(1));

            cv::Mat state(4,1,CV_32F);
            state.at<float>(0) = dets[j].x1;
            state.at<float>(1) = dets[j].y1;
            state.at<float>(2) = dets[j].x2 - dets[j].x1;
            state.at<float>(3) = dets[j].y2 - dets[j].y1;
            t.kf.statePost = state;

            t.bbox = cv::Rect2f(dets[j].x1, dets[j].y1,
                                dets[j].x2 - dets[j].x1,
                                dets[j].y2 - dets[j].y1);
            t.hist = calc_hist(dets[j].roi);
            t.age = 1;
            t.missed = 0;
            t.active = true;

            tracks.push_back(t);
            log_debug("New person appeared: ID=%d\n", t.id);
        }
    }

    tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
                [](const Track& t){
                    if(t.missed > MAX_MISSED) {
                        log_debug("Person disappeared: ID=%d\n", t.id);
                        return true;
                    }
                    return false;
                }), tracks.end());

    return tracks;
}

