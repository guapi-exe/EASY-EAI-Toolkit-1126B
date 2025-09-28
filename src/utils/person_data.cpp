#include <math.h>
#include "person_data.h"
#include "person_detect.h"
#include <string.h>

void reset_person_list() {
    memset(g_person_list, 0, sizeof(g_person_list));
    g_next_person_id = 1;
}

void calc_histogram(const uint8_t *rgb, int img_w, int img_h,
                    BOX_RECT box, float hist[FEATURE_HIST_BIN])
{
    memset(hist, 0, FEATURE_HIST_BIN * sizeof(float));
    int count = 0;

    for (int y = box.top; y < box.bottom; y++) {
        if (y < 0 || y >= img_h) continue;
        for (int x = box.left; x < box.right; x++) {
            if (x < 0 || x >= img_w) continue;

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

int is_same_person(PersonRecord *p, BOX_RECT box, float hist[FEATURE_HIST_BIN], int img_w, int img_h)
{
    // 位移判断
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

    // 颜色匹配
    float d = hist_distance(p->color_hist, hist);
    if (d < 0.25f)
        return 1;

    return 0;
}

// -----------------------------
// 匹配或注册新人员
// 返回值：
// -1 → 新人
// >=0 → 已存在人员ID
// -----------------------------
int match_or_register_person(const uint8_t *rgb, int img_w, int img_h, BOX_RECT box, uint64_t frame_id)
{
    float hist[FEATURE_HIST_BIN];
    calc_histogram(rgb, img_w, img_h, box, hist);

    // 清理超时人员
    for (int i = 0; i < MAX_TRACKED_PERSON; i++) {
        if (g_person_list[i].active &&
            frame_id - g_person_list[i].last_seen_frame > PERSON_TIMEOUT_FRAMES) {
            g_person_list[i].active = 0;
        }
    }

    // 匹配旧人
    for (int i = 0; i < MAX_TRACKED_PERSON; i++) {
        if (!g_person_list[i].active) continue;
        if (is_same_person(&g_person_list[i], box, hist, img_w, img_h)) {
            g_person_list[i].last_seen_frame = frame_id;
            g_person_list[i].box = box;
            return g_person_list[i].id;
        }
    }

    // 注册新人
    for (int i = 0; i < MAX_TRACKED_PERSON; i++) {
        if (!g_person_list[i].active) {
            g_person_list[i].active = 1;
            g_person_list[i].id = g_next_person_id++;
            g_person_list[i].box = box;
            memcpy(g_person_list[i].color_hist, hist, sizeof(hist));
            g_person_list[i].last_seen_frame = frame_id;

            printf("⚡ New person detected: ID=%d at frame %llu\n",
                   g_person_list[i].id, (unsigned long long)frame_id);
            return -1;  // 新人
        }
    }

    // 如果列表满了，也返回新人，但不注册（可选处理）
    return -1;
}