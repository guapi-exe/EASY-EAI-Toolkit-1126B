#include "sort_tracker.h"
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>
extern "C" {
#include "log.h"
}

static std::vector<Track> g_tracks;
static int g_next_id = 1;
static const int MAX_LOST = 30; // 轨迹丢失多少帧删除

static int next_id = 1;
static std::vector<Track> tracks;

static float iou(const Track& t, const Detection& d)
{
    float xx1 = std::max(t.x1, d.x1);
    float yy1 = std::max(t.y1, d.y1);
    float xx2 = std::min(t.x2, d.x2);
    float yy2 = std::min(t.y2, d.y2);
    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter = w * h;
    float area_t = (t.x2 - t.x1) * (t.y2 - t.y1);
    float area_d = (d.x2 - d.x1) * (d.y2 - d.y1);
    return inter / (area_t + area_d - inter + 1e-6f);
}

void sort_init()
{
    tracks.clear();
    next_id = 1;
}

static void predict(Track& t)
{
    t.x1 += t.vx;
    t.y1 += t.vy;
    t.x2 += t.vw;
    t.y2 += t.vh;
    t.age++;
    t.missed++;
}

static void correct(Track& t, const Detection& d)
{
    // 简单 α=0.8 滤波
    float alpha = 0.8f;
    float new_x1 = alpha * t.x1 + (1 - alpha) * d.x1;
    float new_y1 = alpha * t.y1 + (1 - alpha) * d.y1;
    float new_x2 = alpha * t.x2 + (1 - alpha) * d.x2;
    float new_y2 = alpha * t.y2 + (1 - alpha) * d.y2;

    t.vx = new_x1 - t.x1;
    t.vy = new_y1 - t.y1;
    t.vw = new_x2 - t.x2;
    t.vh = new_y2 - t.y2;

    t.x1 = new_x1;
    t.y1 = new_y1;
    t.x2 = new_x2;
    t.y2 = new_y2;
    t.missed = 0;
    t.active = true;
}

std::vector<Track> sort_update(const std::vector<Detection>& dets)
{
    // 1️ 预测所有已有 track
    for (auto& t : tracks) predict(t);

    // 2️ 关联检测结果
    std::vector<int> det_assigned(dets.size(), -1);

    for (size_t i = 0; i < dets.size(); i++) {
        float best_iou = 0.0f;
        int best_j = -1;
        for (size_t j = 0; j < tracks.size(); j++) {
            float iou_score = iou(tracks[j], dets[i]);
            if (iou_score > 0.3f && iou_score > best_iou) {
                best_iou = iou_score;
                best_j = j;
            }
        }
        if (best_j != -1) {
            correct(tracks[best_j], dets[i]);
            det_assigned[i] = best_j;
        }
    }

    // 3️对未匹配的检测，创建新 track
    for (size_t i = 0; i < dets.size(); i++) {
        if (det_assigned[i] == -1) {
            Track t;
            t.id = next_id++;
            t.x1 = dets[i].x1;
            t.y1 = dets[i].y1;
            t.x2 = dets[i].x2;
            t.y2 = dets[i].y2;
            t.vx = t.vy = t.vw = t.vh = 0;
            t.age = 1;
            t.missed = 0;
            t.active = true;
            tracks.push_back(t);

            log_debug("New person appeared: ID=%d\n", t.id);
        }
    }

    tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
        [](const Track& t){ return t.missed > MAX_LOST; }), tracks.end());

    return tracks;
}
