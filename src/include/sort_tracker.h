#ifndef SORT_TRACKER_H
#define SORT_TRACKER_H

#include <vector>
#include <opencv2/opencv.hpp>

typedef struct {
    float x1, y1, x2, y2;
    float score;
    int class_id;
} Detection;

typedef struct {
    int id;
    cv::Rect2f bbox;
    int lost_frames;
    cv::KalmanFilter kf;
} Track;

void sort_init();
void sort_update(const std::vector<Detection> &detections);
const std::vector<Track>& sort_get_tracks();

#endif
