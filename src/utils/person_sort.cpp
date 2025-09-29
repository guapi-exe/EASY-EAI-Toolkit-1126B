#include "sort_tracker.h"
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>

static std::vector<Track> g_tracks;
static int g_next_id = 1;
static const int MAX_LOST = 30; // 轨迹丢失多少帧删除

void sort_init() {
    g_tracks.clear();
    g_next_id = 1;
}

static float iou(const cv::Rect2f &a, const cv::Rect2f &b) {
    float inter = (a & b).area();
    float uni = a.area() + b.area() - inter;
    return uni > 0 ? inter / uni : 0;
}

void sort_update(const std::vector<Detection> &detections) {
    // 1. 所有track预测
    for (auto &t : g_tracks) {
        cv::Mat pred = t.kf.predict();
        t.bbox.x = pred.at<float>(0);
        t.bbox.y = pred.at<float>(1);
    }

    // 2. 匹配
    std::vector<int> det_matched(detections.size(), -1);
    for (int i = 0; i < g_tracks.size(); i++) {
        float best_iou = 0;
        int best_det = -1;
        for (int j = 0; j < detections.size(); j++) {
            if (det_matched[j] != -1) continue;
            cv::Rect2f det_box(detections[j].x1, detections[j].y1,
                               detections[j].x2 - detections[j].x1,
                               detections[j].y2 - detections[j].y1);
            float ov = iou(g_tracks[i].bbox, det_box);
            if (ov > 0.3 && ov > best_iou) {
                best_iou = ov;
                best_det = j;
            }
        }
        if (best_det != -1) {
            // 更新track
            const auto &det = detections[best_det];
            cv::Rect2f det_box(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);
            cv::Mat meas(4, 1, CV_32F);
            meas.at<float>(0) = det_box.x;
            meas.at<float>(1) = det_box.y;
            meas.at<float>(2) = det_box.width;
            meas.at<float>(3) = det_box.height;
            g_tracks[i].kf.correct(meas);
            g_tracks[i].bbox = det_box;
            g_tracks[i].lost_frames = 0;
            det_matched[best_det] = i;
        } else {
            g_tracks[i].lost_frames++;
        }
    }

    // 3. 新检测 → 新track
    for (int j = 0; j < detections.size(); j++) {
        if (det_matched[j] != -1) continue;
        Track t;
        t.id = g_next_id++;
        t.bbox = cv::Rect2f(detections[j].x1, detections[j].y1,
                            detections[j].x2 - detections[j].x1,
                            detections[j].y2 - detections[j].y1);
        t.lost_frames = 0;

        t.kf = cv::KalmanFilter(4, 4);
        t.kf.transitionMatrix = (cv::Mat_<float>(4,4) <<
            1,0,1,0,
            0,1,0,1,
            0,0,1,0,
            0,0,0,1);
        cv::setIdentity(t.kf.measurementMatrix);
        cv::setIdentity(t.kf.processNoiseCov, cv::Scalar::all(1e-2));
        cv::setIdentity(t.kf.measurementNoiseCov, cv::Scalar::all(1e-1));
        cv::setIdentity(t.kf.errorCovPost, cv::Scalar::all(1));
        t.kf.statePost.at<float>(0) = t.bbox.x;
        t.kf.statePost.at<float>(1) = t.bbox.y;
        t.kf.statePost.at<float>(2) = 0;
        t.kf.statePost.at<float>(3) = 0;

        g_tracks.push_back(t);
    }

    // 4. 清理丢失轨迹
    g_tracks.erase(std::remove_if(g_tracks.begin(), g_tracks.end(),
        [](const Track &t){ return t.lost_frames > MAX_LOST; }), g_tracks.end());
}

const std::vector<Track>& sort_get_tracks() {
    return g_tracks;
}
