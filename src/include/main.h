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
#define PERSON_DETECT_THRESH    0.7f   
#define FACE_CLARITY_THRESH     50.0    
#define FACE_AREA_RATIO_MIN     0.05f
#define PERSON_ROI_MAX_WIDTH    640    

// 人脸姿态检测阈值（加强检测，避免侧脸和背面）
#define FACE_YAW_THRESH         0.2f   
#define FACE_ROLL_THRESH        15.0f   
#define FACE_PITCH_THRESH       0.3f   
#define FACE_EYE_DISTANCE_MIN   0.15f  
#define FACE_SYMMETRY_THRESH    0.15f 