#ifndef _RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_
#define _RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_

#include <stdint.h>
#include <vector>

#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 64
#define OBJ_CLASS_NUM     1
#define NMS_THRESH        0.45
#define BOX_THRESH        0.25
#define PROP_BOX_SIZE     (5+OBJ_CLASS_NUM)

typedef struct _BOX_RECT
{
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT;

typedef struct __detect_result_t
{
    char name[OBJ_NAME_MAX_SIZE];
    int class_index;
    BOX_RECT box;
    float prop;
} detect_result_t;

typedef struct _detect_result_group_t
{
    int id;
    int count;
    detect_result_t results[OBJ_NUMB_MAX_SIZE];
} detect_result_group_t;

#ifdef __cplusplus
extern "C" {
#endif

int person_post_process(int8_t* input0, int8_t* input1, int8_t* input2,
                        int model_in_w, int model_in_h,
                        float conf_threshold, float nms_threshold,
                        std::vector<int>& qnt_zps,
                        std::vector<float>& qnt_scales,
                        detect_result_group_t* group);

void deinitPostProcess();

#ifdef __cplusplus
}
#endif
#endif //_RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_
