#include <math.h>
#include "person_data.h"
#include "person_detect.h"

// 简单颜色直方图（灰度）
void calc_histogram(const uint8_t *rgb, int img_w, int img_h,
                    BOX_RECT box, float hist[FEATURE_HIST_BIN])
{
    memset(hist, 0, FEATURE_HIST_BIN * sizeof(float));
    int count = 0;

    for (int y = box.top; y < box.bottom; y++) {
        for (int x = box.left; x < box.right; x++) {
            int idx = (y * img_w + x) * 3;
            uint8_t r = rgb[idx + 0];
            uint8_t g = rgb[idx + 1];
            uint8_t b = rgb[idx + 2];
            uint8_t gray = (uint8_t)((r*30 + g*59 + b*11)/100);
            int bin = gray * FEATURE_HIST_BIN / 256;
            hist[bin] += 1;
            count++;
        }
    }

    if (count > 0) {
        for (int i = 0; i < FEATURE_HIST_BIN; i++)
            hist[i] /= (float)count;
    }
}

float hist_distance(const float *a, const float *b)
{
    float d = 0;
    for (int i = 0; i < FEATURE_HIST_BIN; i++)
        d += fabsf(a[i] - b[i]);
    return d;
}

int is_same_person(PersonRecord *p, BOX_RECT box, float hist[FEATURE_HIST_BIN])
{
    // 位置差异
    int cx1 = (p->box.left + p->box.right)/2;
    int cy1 = (p->box.top + p->box.bottom)/2;
    int cx2 = (box.left + box.right)/2;
    int cy2 = (box.top + box.bottom)/2;

    int dx = abs(cx1 - cx2);
    int dy = abs(cy1 - cy2);

    int pw = p->box.right - p->box.left;
    int ph = p->box.bottom - p->box.top;

    if (dx > pw || dy > ph)
        return 0;

    // 颜色差异
    float d = hist_distance(p->color_hist, hist);
    if (d < 0.25f)  // 阈值可调
        return 1;

    return 0;
}
