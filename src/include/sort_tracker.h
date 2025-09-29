#ifndef PERSON_SORT_H
#define PERSON_SORT_H

#include <vector>
#include <cstdint>

struct Detection {
    float x1, y1, x2, y2;  // bbox
    float score;
};

struct Track {
    int id;
    float x1, y1, x2, y2;
    float vx, vy, vw, vh;  // 速度分量
    int age;              // 存活帧数
    int missed;           // 丢失帧数
    bool active;
};

void sort_init();
std::vector<Track> sort_update(const std::vector<Detection>& dets);

#endif
