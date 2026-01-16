#ifndef FACE_DETECT_RETIAN_H
#define FACE_DETECT_RETIAN_H

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "rknn_api.h"

// RetinaFace detection result structure
class RetinaFaceResult {
public:
    RetinaFaceResult() {
        landmarks.resize(5);
    }
    ~RetinaFaceResult() {}

    bool operator<(const RetinaFaceResult &t) const {
        return score < t.score;
    }

    bool operator>(const RetinaFaceResult &t) const {
        return score > t.score;
    }

    cv::Rect_<float> box;              // Face bounding box
    std::vector<cv::Point2f> landmarks; // 5 facial landmarks
    float score;                        // Detection confidence

    void print() {
        printf("Face box(x1y1x2y2): %.2f %.2f %.2f %.2f, score: %.3f\n", 
               box.x, box.y, box.br().x, box.br().y, score);
        printf("Landmarks: ");
        for (size_t i = 0; i < landmarks.size(); i++) {
            printf("(%.2f, %.2f) ", landmarks[i].x, landmarks[i].y);
        }
        printf("\n");
    }
};

// Model configuration types
enum RetinaFaceModelType {
    RETINAFACE_MODEL,  // Original RetinaFace
    SLIM_MODEL,        // Slim version
    RFB_MODEL          // RFB version
};

// Model configuration structure
struct RetinaFaceConfig {
    RetinaFaceModelType model_type;
    std::vector<std::vector<int>> min_sizes;
    std::vector<int> steps;
    float variance[2];  // [0.1, 0.2]
    int input_height;
    int input_width;
};

/* 
 * RetinaFace detection initialization
 * ctx: Output parameter, rknn_context handle
 * model_path: Input parameter, path to .rknn model file
 * config: Input parameter, model configuration
 * Returns: 0 on success, -1 on failure
 */
int face_detect_retian_init(rknn_context *ctx, const char *model_path, RetinaFaceConfig *config);

/* 
 * RetinaFace detection inference
 * ctx: Input parameter, rknn_context handle
 * input_image: Input parameter, OpenCV Mat format image (BGR)
 * results: Output parameter, detected faces with landmarks
 * conf_thresh: Confidence threshold (default 0.5)
 * nms_thresh: NMS threshold (default 0.4)
 * keep_top_k: Maximum number of faces to keep (default 10)
 * Returns: Number of detected faces
 */
int face_detect_retian_run(rknn_context ctx, 
                           cv::Mat &input_image, 
                           std::vector<RetinaFaceResult> &results,
                           float conf_thresh = 0.5f,
                           float nms_thresh = 0.4f,
                           int keep_top_k = 10);

/* 
 * RetinaFace detection release
 * ctx: Input parameter, rknn_context handle
 * Returns: 0 on success
 */
int face_detect_retian_release(rknn_context ctx);

/* 
 * Helper function: Get predefined model configuration
 * model_type: Model type (RETINAFACE_MODEL, SLIM_MODEL, RFB_MODEL)
 * input_h: Model input height
 * input_w: Model input width
 * Returns: Model configuration
 */
RetinaFaceConfig get_retian_config(RetinaFaceModelType model_type, int input_h, int input_w);

#endif // FACE_DETECT_RETIAN_H
