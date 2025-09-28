#ifndef PERSON_DATA_H
#define PERSON_DATA_H

#define MAX_TRACKED_PERSON 32
#define FEATURE_HIST_BIN   16
#define PERSON_TIMEOUT_FRAMES 50

#include "person_detect.h"

typedef struct
{
    int id;                 
    BOX_RECT box;           
    float color_hist[FEATURE_HIST_BIN];
    int last_seen_frame;    // 最近出现的帧编号
    int active;             
} PersonRecord;

static PersonRecord g_person_list[MAX_TRACKED_PERSON];
static int g_next_person_id = 1;
int is_same_person(PersonRecord *p, BOX_RECT box, float hist[FEATURE_HIST_BIN]);
void calc_histogram(const uint8_t *rgb, int img_w, int img_h,BOX_RECT box, float hist[FEATURE_HIST_BIN]);
void reset_person_list();
int match_or_register_person(const uint8_t *rgb, int img_w, int img_h, BOX_RECT box, uint64_t frame_id);
float hist_distance(const float *a, const float *b)

#endif