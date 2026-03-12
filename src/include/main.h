#pragma once

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#define CAMERA_WIDTH    3840
#define CAMERA_HEIGHT   2160
#define IMAGE_WIDTH     1280
#define IMAGE_HEIGHT    720
#define CAMERA_INDEX_1  11
#define CAMERA_INDEX_2  51
#define DEFAULT_CAMERA_NUMBER 1
#define CAMERA_FORMAT   RK_FORMAT_BGR_888  
#define IMGRATIO        3
#define IMAGE_SIZE      (CAMERA_WIDTH*CAMERA_HEIGHT*IMGRATIO)
#define MAX_MISSED      45

#define PERSON_MODEL_PATH   "person_detect.model"
#define FACE_MODEL_PATH     "face_detect.model"

#define RETIAN_MODEL_TYPE   0 
#define RETIAN_INPUT_H      480
#define RETIAN_INPUT_W      640
#define RETIAN_CONF_THRESH  0.7f
#define RETIAN_NMS_THRESH   0.4f

// 抓拍与性能调优参数
#define CAPTURE_MIN_CLARITY              80.0
#define CAPTURE_MAX_MOTION_RATIO         0.020f
#define CAPTURE_PERSON_DETECT_INTERVAL   3
#define CAPTURE_FACE_DETECT_INTERVAL    2
#define CAPTURE_FACE_INPUT_MAX_WIDTH    512
#define CAPTURE_MIN_FACE_SCORE          0.55f
#define CAPTURE_HEADSHOT_EXPAND_RATIO   1.80f
#define CAPTURE_HEADSHOT_DOWN_SHIFT     0.20f
#define CAPTURE_FOCUS_SCALE_FACTOR      3
#define CAPTURE_MIN_AREA_RATIO          0.035f
#define CAPTURE_APPROACH_RATIO_POS      0.08f
#define CAPTURE_APPROACH_RATIO_NEG     -0.08f
#define CAPTURE_MIN_TRACK_HITS          3
#define CAPTURE_REQUIRE_APPROACH        0
#define CAPTURE_MAX_YAW                 0.90f