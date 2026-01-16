#ifndef FACE_DETECT_SCRFD_H
#define FACE_DETECT_SCRFD_H

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "rknn_api.h"

// SCRFD detection result structure
class SCRFDResult {
public:
    SCRFDResult() {}
    ~SCRFDResult() {}

    bool operator<(const SCRFDResult &t) const {
        return score < t.score;
    }

    bool operator>(const SCRFDResult &t) const {
        return score > t.score;
    }

    cv::Rect_<float> box;              // Face bounding box
    float score;                        // Detection confidence

    void print() {
        printf("Face box(x1y1x2y2): %.2f %.2f %.2f %.2f, score: %.3f\n", 
               box.x, box.y, box.br().x, box.br().y, score);
    }
};

// SCRFD model configuration
struct SCRFDConfig {
    int input_height;
    int input_width;
    std::vector<int> strides;          // Feature pyramid strides (e.g., {8, 16, 32})
    float conf_thresh;                 // Confidence threshold
    float nms_thresh;                  // NMS threshold
};

/* 
 * SCRFD detection initialization
 * ctx: Output parameter, rknn_context handle
 * model_path: Input parameter, path to .rknn model file
 * config: Input parameter, model configuration
 * Returns: 0 on success, -1 on failure
 */
int face_detect_scrfd_init(rknn_context *ctx, const char *model_path, SCRFDConfig *config);

/* 
 * SCRFD detection inference
 * ctx: Input parameter, rknn_context handle
 * input_image: Input parameter, OpenCV Mat format image (BGR)
 * results: Output parameter, detected faces
 * Returns: Number of detected faces
 */
int face_detect_scrfd_run(rknn_context ctx, 
                          cv::Mat &input_image, 
                          std::vector<SCRFDResult> &results);

/* 
 * SCRFD detection release
 * ctx: Input parameter, rknn_context handle
 * Returns: 0 on success
 */
int face_detect_scrfd_release(rknn_context ctx);

/* 
 * Helper function: Get default SCRFD configuration
 * input_h: Model input height
 * input_w: Model input width
 * Returns: Default SCRFD configuration
 */
SCRFDConfig get_scrfd_config(int input_h, int input_w);

#endif // FACE_DETECT_SCRFD_H
