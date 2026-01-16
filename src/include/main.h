#pragma once

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#define CAMERA_WIDTH    2688
#define CAMERA_HEIGHT   1520
#define IMAGE_WIDTH     960
#define IMAGE_HEIGHT    540
#define CAMERA_INDEX_1  23
#define CAMERA_INDEX_2  51
#define IMGRATIO        3
#define IMAGE_SIZE      (CAMERA_WIDTH*CAMERA_HEIGHT*IMGRATIO)
#define MAX_MISSED      30

// 模型路径配置
#define PERSON_MODEL_PATH   "person_detect.model"
#define FACE_MODEL_PATH     "retinaface_mobilenet0.25_480x640.rknn"

// RetinaFace 配置
#define RETIAN_MODEL_TYPE   0 
#define RETIAN_INPUT_H      480
#define RETIAN_INPUT_W      640
#define RETIAN_CONF_THRESH  0.5f
#define RETIAN_NMS_THRESH   0.4f

// 性能优化配置
#define PERSON_DETECT_THRESH    0.7f    // 人员检测置信度阈值
#define FACE_CLARITY_THRESH     50.0    // 人脸清晰度阈值
#define FACE_AREA_RATIO_MIN     0.05f   // 最小人脸面积比例
#define PERSON_ROI_MAX_WIDTH    640     // 人员ROI最大宽度（减少人脸检测耗时）