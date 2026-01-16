#include "face_detect_retian.h"
#include "postprocess.h"
#include <math.h>
#include <string.h>
#include <algorithm>

static RetinaFaceConfig g_config;
static std::vector<std::vector<float>> g_priors;

// Helper function: Generate prior boxes (anchor boxes)
static void generate_priors(const RetinaFaceConfig &cfg, std::vector<std::vector<float>> &priors) {
    priors.clear();
    
    for (size_t k = 0; k < cfg.steps.size(); k++) {
        int step = cfg.steps[k];
        int feature_h = (int)ceil((float)cfg.input_height / step);
        int feature_w = (int)ceil((float)cfg.input_width / step);
        
        for (int i = 0; i < feature_h; i++) {
            for (int j = 0; j < feature_w; j++) {
                for (size_t m = 0; m < cfg.min_sizes[k].size(); m++) {
                    int min_size = cfg.min_sizes[k][m];
                    float s_kx = (float)min_size / cfg.input_width;
                    float s_ky = (float)min_size / cfg.input_height;
                    float dense_cx = (j + 0.5f) * step / cfg.input_width;
                    float dense_cy = (i + 0.5f) * step / cfg.input_height;
                    
                    std::vector<float> prior = {dense_cx, dense_cy, s_kx, s_ky};
                    priors.push_back(prior);
                }
            }
        }
    }
}

// Get predefined model configuration
RetinaFaceConfig get_retian_config(RetinaFaceModelType model_type, int input_h, int input_w) {
    RetinaFaceConfig cfg;
    cfg.model_type = model_type;
    cfg.input_height = input_h;
    cfg.input_width = input_w;
    cfg.variance[0] = 0.1f;
    cfg.variance[1] = 0.2f;
    
    switch (model_type) {
        case RETINAFACE_MODEL:
            cfg.min_sizes = {{10, 20}, {32, 64}, {128, 256}};
            cfg.steps = {8, 16, 32};
            break;
        case SLIM_MODEL:
        case RFB_MODEL:
            cfg.min_sizes = {{10, 16, 24}, {32, 48}, {64, 96}, {128, 192, 256}};
            cfg.steps = {8, 16, 32, 64};
            break;
        default:
            cfg.min_sizes = {{10, 20}, {32, 64}, {128, 256}};
            cfg.steps = {8, 16, 32};
            break;
    }
    
    return cfg;
}

int face_detect_retian_init(rknn_context *ctx, const char *model_path, RetinaFaceConfig *config) {
    if (!ctx || !model_path || !config) {
        printf("Invalid parameters for face_detect_retian_init\n");
        return -1;
    }
    
    // Save config
    g_config = *config;
    
    // Generate prior boxes
    generate_priors(g_config, g_priors);
    printf("Generated %zu prior boxes\n", g_priors.size());
    
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
    
    printf("RetinaFace model initialized successfully\n");
    return 0;
}

int face_detect_retian_release(rknn_context ctx) {
    if (ctx == 0) {
        return -1;
    }
    
    g_priors.clear();
    rknn_destroy(ctx);
    printf("RetinaFace model released\n");
    return 0;
}

int face_detect_retian_run(rknn_context ctx, 
                           cv::Mat &input_image, 
                           std::vector<RetinaFaceResult> &results,
                           float conf_thresh,
                           float nms_thresh,
                           int keep_top_k) {
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
    
    // Expected: 1 input, 3 outputs (loc, conf, landmarks)
    if (io_num.n_input != 1 || io_num.n_output != 3) {
        printf("Model expects 1 input and 3 outputs, got %d inputs and %d outputs\n", 
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
    rknn_output outputs[3];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < 3; i++) {
        outputs[i].index = i;
        outputs[i].want_float = 1;
        outputs[i].is_prealloc = 0;
    }
    
    ret = rknn_outputs_get(ctx, 3, outputs, NULL);
    if (ret < 0) {
        printf("rknn_outputs_get failed: %d\n", ret);
        return 0;
    }
    
    // Parse outputs: loc, conf, landms
    float *loc_data = (float *)outputs[0].buf;
    float *conf_data = (float *)outputs[1].buf;
    float *landms_data = (float *)outputs[2].buf;
    
    int num_priors = g_priors.size();
    
    // Decode boxes and landmarks
    std::vector<RetinaFaceResult> candidates;
    
    for (int i = 0; i < num_priors; i++) {
        // Check confidence
        float conf = conf_data[i * 2 + 1];  // Index 1 is face class
        if (conf < conf_thresh) {
            continue;
        }
        
        RetinaFaceResult result;
        result.score = conf;
        
        // Decode box
        float cx = g_priors[i][0] + loc_data[i * 4 + 0] * g_config.variance[0] * g_priors[i][2];
        float cy = g_priors[i][1] + loc_data[i * 4 + 1] * g_config.variance[0] * g_priors[i][3];
        float w = g_priors[i][2] * exp(loc_data[i * 4 + 2] * g_config.variance[1]);
        float h = g_priors[i][3] * exp(loc_data[i * 4 + 3] * g_config.variance[1]);
        
        float x1 = (cx - w * 0.5f) * g_config.input_width;
        float y1 = (cy - h * 0.5f) * g_config.input_height;
        float x2 = (cx + w * 0.5f) * g_config.input_width;
        float y2 = (cy + h * 0.5f) * g_config.input_height;
        
        result.box = cv::Rect_<float>(x1, y1, x2 - x1, y2 - y1);
        
        // Decode landmarks
        for (int j = 0; j < 5; j++) {
            float lm_x = g_priors[i][0] + landms_data[i * 10 + j * 2] * g_config.variance[0] * g_priors[i][2];
            float lm_y = g_priors[i][1] + landms_data[i * 10 + j * 2 + 1] * g_config.variance[0] * g_priors[i][3];
            result.landmarks[j].x = lm_x * g_config.input_width;
            result.landmarks[j].y = lm_y * g_config.input_height;
        }
        
        candidates.push_back(result);
    }
    
    // Release outputs
    rknn_outputs_release(ctx, 3, outputs);
    
    if (candidates.empty()) {
        return 0;
    }
    
    // NMS
    std::sort(candidates.begin(), candidates.end(), std::greater<RetinaFaceResult>());
    
    std::vector<bool> suppressed(candidates.size(), false);
    for (size_t i = 0; i < candidates.size(); i++) {
        if (suppressed[i]) continue;
        
        results.push_back(candidates[i]);
        
        if ((int)results.size() >= keep_top_k) {
            break;
        }
        
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
            
            if (iou > nms_thresh) {
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
        
        for (auto &lm : res.landmarks) {
            lm.x *= scale_x;
            lm.y *= scale_y;
        }
    }
    
    return (int)results.size();
}
