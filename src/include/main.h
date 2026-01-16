#pragma once

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#define CAMERA_WIDTH    2688
#define CAMERA_HEIGHT   1520
#define IMAGE_WIDTH     1280
#define IMAGE_HEIGHT    720
#define CAMERA_INDEX_1  23
#define CAMERA_INDEX_2  51
#define IMGRATIO        3
#define IMAGE_SIZE      (CAMERA_WIDTH*CAMERA_HEIGHT*IMGRATIO)
#define MAX_MISSED      30

#define PERSON_MODEL_PATH   "person_detect.model"
#define FACE_MODEL_PATH     "RFB_480x640.rknn"

#define RETIAN_MODEL_TYPE   0 
#define RETIAN_INPUT_H      480
#define RETIAN_INPUT_W      640
#define RETIAN_CONF_THRESH  0.7f
#define RETIAN_NMS_THRESH   0.4f