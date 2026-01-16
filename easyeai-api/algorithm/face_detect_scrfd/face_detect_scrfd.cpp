#include "face_detect_scrfd.h"
#include "postprocess.h"
#include <math.h>
#include <string.h>
#include <algorithm>

static SCRFDConfig g_config;
static std::vector<std::vector<cv::Point2f>> g_anchor_centers;

// Helper function: Sigmoid activation
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Generate anchor centers for each stride
static void generate_anchors(const SCRFDConfig &cfg, std::vector<std::vector<cv::Point2f>> &anchors) {
    anchors.clear();
    anchors.resize(cfg.strides.size());
    
    for (size_t i = 0; i < cfg.strides.size(); i++) {
        int stride = cfg.strides[i];
        int feat_h = cfg.input_height / stride;
        int feat_w = cfg.input_width / stride;
        
        for (int y = 0; y < feat_h; y++) {
            for (int x = 0; x < feat_w; x++) {
                float cx = (x + 0.5f) * stride;
                float cy = (y + 0.5f) * stride;
                anchors[i].push_back(cv::Point2f(cx, cy));
            }
        }
    }
}

// Get default SCRFD configuration
SCRFDConfig get_scrfd_config(int input_h, int input_w) {
    SCRFDConfig cfg;
    cfg.input_height = input_h;
    cfg.input_width = input_w;
    cfg.strides = {8, 16, 32};  // Default strides for SCRFD-1G
    cfg.conf_thresh = 0.6f;
    cfg.nms_thresh = 0.4f;
    return cfg;
}

int face_detect_scrfd_init(rknn_context *ctx, const char *model_path, SCRFDConfig *config) {
    if (!ctx || !model_path || !config) {
        printf("Invalid parameters for face_detect_scrfd_init\n");
        return -1;
    }
    
    // Save config
    g_config = *config;
    
    // Generate anchor centers
    generate_anchors(g_config, g_anchor_centers);
    printf("Generated anchors for strides: ");
    for (auto stride : g_config.strides) {
        printf("%d ", stride);
    }
    printf("\n");
    
    // Load RKNN model
    FILE *fp = fopen(model_path, "rb");
    if (!fp) {
        printf("Failed to open model file: %s\n", model_path);
        return -1;
    }
    
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    void *model_data = malloc(model_len);
    if (!model_data) {
        printf("Failed to allocate memory for model\n");
        fclose(fp);
        return -1;
    }
    
    if (fread(model_data, 1, model_len, fp) != (size_t)model_len) {
        printf("Failed to read model file\n");
        free(model_data);
        fclose(fp);
        return -1;
    }
    fclose(fp);
    
    // Initialize RKNN
    int ret = rknn_init(ctx, model_data, model_len, 0, NULL);
    free(model_data);
    
    if (ret < 0) {
        printf("rknn_init failed: %d\n", ret);
        return -1;
    }
    
    printf("SCRFD model initialized successfully\n");
    return 0;
}

int face_detect_scrfd_release(rknn_context ctx) {
    if (ctx == 0) {
        return -1;
    }
    
    g_anchor_centers.clear();
    rknn_destroy(ctx);
    printf("SCRFD model released\n");
    return 0;
}

int face_detect_scrfd_run(rknn_context ctx, 
                          cv::Mat &input_image, 
                          std::vector<SCRFDResult> &results) {
    results.clear();
    
    if (input_image.empty()) {
        printf("Input image is empty\n");
        return 0;
    }
    
    // Get model input/output info
    rknn_input_output_num io_num;
    int ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query failed: %d\n", ret);
        return 0;
    }
    
    // SCRFD-1G: 1 input, 6 outputs (3 scores + 3 bboxes)
    if (io_num.n_input != 1 || io_num.n_output != 6) {
        printf("Model expects 1 input and 6 outputs, got %d inputs and %d outputs\n", 
               io_num.n_input, io_num.n_output);
        return 0;
    }
    
    // Preprocess: resize and normalize
    cv::Mat resized;
    float scale_x = (float)input_image.cols / g_config.input_width;
    float scale_y = (float)input_image.rows / g_config.input_height;
    
    cv::resize(input_image, resized, cv::Size(g_config.input_width, g_config.input_height));
    
    // Normalize: (pixel - 127.5) / 128.0
    cv::Mat normalized;
    resized.convertTo(normalized, CV_32FC3);
    normalized = (normalized - 127.5) / 128.0;
    
    // Prepare input
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = g_config.input_width * g_config.input_height * 3;
    inputs[0].buf = resized.data;
    
    ret = rknn_inputs_set(ctx, 1, inputs);
    if (ret < 0) {
        printf("rknn_inputs_set failed: %d\n", ret);
        return 0;
    }
    
    // Run inference
    ret = rknn_run(ctx, NULL);
    if (ret < 0) {
        printf("rknn_run failed: %d\n", ret);
        return 0;
    }
    
    // Get outputs
    rknn_output outputs[6];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < 6; i++) {
        outputs[i].index = i;
        outputs[i].want_float = 1;
        outputs[i].is_prealloc = 0;
    }
    
    ret = rknn_outputs_get(ctx, 6, outputs, NULL);
    if (ret < 0) {
        printf("rknn_outputs_get failed: %d\n", ret);
        return 0;
    }
    
    // Parse outputs: score_8, score_16, score_32, bbox_8, bbox_16, bbox_32
    float *score_data[3] = {
        (float *)outputs[0].buf,  // score_8  [12800,1]
        (float *)outputs[1].buf,  // score_16 [3200,1]
        (float *)outputs[2].buf   // score_32 [800,1]
    };
    
    float *bbox_data[3] = {
        (float *)outputs[3].buf,  // bbox_8  [12800,4]
        (float *)outputs[4].buf,  // bbox_16 [3200,4]
        (float *)outputs[5].buf   // bbox_32 [800,4]
    };
    
    // Decode each stride
    std::vector<SCRFDResult> candidates;
    
    for (size_t stride_idx = 0; stride_idx < g_config.strides.size(); stride_idx++) {
        int stride = g_config.strides[stride_idx];
        int num_anchors = g_anchor_centers[stride_idx].size();
        
        for (int i = 0; i < num_anchors; i++) {
            // Get score (sigmoid)
            float score = sigmoid(score_data[stride_idx][i]);
            
            if (score < g_config.conf_thresh) {
                continue;
            }
            
            // Get bbox (ltrb format)
            float *bbox = &bbox_data[stride_idx][i * 4];
            float l = bbox[0];
            float t = bbox[1];
            float r = bbox[2];
            float b = bbox[3];
            
            // Decode bbox (anchor-free)
            cv::Point2f center = g_anchor_centers[stride_idx][i];
            float x1 = center.x - l * stride;
            float y1 = center.y - t * stride;
            float x2 = center.x + r * stride;
            float y2 = center.y + b * stride;
            
            // Clip to image boundaries
            x1 = std::max(0.0f, std::min((float)g_config.input_width, x1));
            y1 = std::max(0.0f, std::min((float)g_config.input_height, y1));
            x2 = std::max(0.0f, std::min((float)g_config.input_width, x2));
            y2 = std::max(0.0f, std::min((float)g_config.input_height, y2));
            
            if (x2 <= x1 || y2 <= y1) {
                continue;
            }
            
            SCRFDResult result;
            result.box = cv::Rect_<float>(x1, y1, x2 - x1, y2 - y1);
            result.score = score;
            
            candidates.push_back(result);
        }
    }
    
    // Release outputs
    rknn_outputs_release(ctx, 6, outputs);
    
    if (candidates.empty()) {
        return 0;
    }
    
    // NMS
    std::sort(candidates.begin(), candidates.end(), std::greater<SCRFDResult>());
    
    std::vector<bool> suppressed(candidates.size(), false);
    for (size_t i = 0; i < candidates.size(); i++) {
        if (suppressed[i]) continue;
        
        results.push_back(candidates[i]);
        
        cv::Rect_<float> &box1 = candidates[i].box;
        float area1 = box1.width * box1.height;
        
        for (size_t j = i + 1; j < candidates.size(); j++) {
            if (suppressed[j]) continue;
            
            cv::Rect_<float> &box2 = candidates[j].box;
            float area2 = box2.width * box2.height;
            
            float inter_x1 = std::max(box1.x, box2.x);
            float inter_y1 = std::max(box1.y, box2.y);
            float inter_x2 = std::min(box1.br().x, box2.br().x);
            float inter_y2 = std::min(box1.br().y, box2.br().y);
            
            float inter_w = std::max(0.0f, inter_x2 - inter_x1);
            float inter_h = std::max(0.0f, inter_y2 - inter_y1);
            float inter_area = inter_w * inter_h;
            
            float union_area = area1 + area2 - inter_area;
            float iou = inter_area / union_area;
            
            if (iou > g_config.nms_thresh) {
                suppressed[j] = true;
            }
        }
    }
    
    // Scale back to original image size
    for (auto &res : results) {
        res.box.x *= scale_x;
        res.box.y *= scale_y;
        res.box.width *= scale_x;
        res.box.height *= scale_y;
    }
    
    return (int)results.size();
}
