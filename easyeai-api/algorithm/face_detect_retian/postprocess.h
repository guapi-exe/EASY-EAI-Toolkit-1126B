#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <stdint.h>

// Common detection result structure (compatible with yolov5 style)
typedef struct {
    int left;
    int top;
    int right;
    int bottom;
} box_rect;

typedef struct {
    box_rect box;
    float prop;
    int cls_id;
    char name[16];
} detect_result_t;

typedef struct {
    int count;
    detect_result_t results[128];
} detect_result_group_t;

#endif // POSTPROCESS_H
