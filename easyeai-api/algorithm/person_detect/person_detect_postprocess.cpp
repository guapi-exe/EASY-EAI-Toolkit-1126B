/**
 * person_detect_postprocess.cpp - YOLO后处理模块
 * 
 * 纯C++11实现，兼容OpenCV 4.6
 */

#include "postprocess.h"
#include <vector>
#include <set>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>

// YOLO anchor 配置
static const int anchor0[] = {10, 13, 16, 30, 33, 23};       // stride 8
static const int anchor1[] = {30, 61, 62, 45, 59, 119};      // stride 16
static const int anchor2[] = {116, 90, 156, 198, 373, 326};  // stride 32

// 类别标签
static const char labels[][30] = {"person"};

// ========== 辅助函数 ==========

static inline float clip(float val, float min_val, float max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

static inline int clamp(float val, int min_val, int max_val) {
    return (int)clip(val, (float)min_val, (float)max_val);
}

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float unsigmoid(float y) {
    return -logf(1.0f / y - 1.0f);
}

static inline float deqnt_affine_to_f32(float scale, int8_t qnt_val, int zero_point) {
    return ((float)qnt_val - (float)zero_point) * scale;
}

static inline int8_t qnt_f32_to_affine(float val, int zero_point, float scale) {
    return (int8_t)clip(val / scale + (float)zero_point, -128.0f, 127.0f);
}

static float CalculateOverlap(float x1_min, float y1_min, float x1_max, float y1_max,
                               float x2_min, float y2_min, float x2_max, float y2_max) {
    float inter_x_min = std::max(x1_min, x2_min);
    float inter_y_min = std::max(y1_min, y2_min);
    float inter_x_max = std::min(x1_max, x2_max);
    float inter_y_max = std::min(y1_max, y2_max);
    
    float inter_w = std::max(0.0f, inter_x_max - inter_x_min + 1.0f);
    float inter_h = std::max(0.0f, inter_y_max - inter_y_min + 1.0f);
    float inter_area = inter_w * inter_h;
    
    float area1 = (x1_max - x1_min + 1.0f) * (y1_max - y1_min + 1.0f);
    float area2 = (x2_max - x2_min + 1.0f) * (y2_max - y2_min + 1.0f);
    float union_area = area1 + area2 - inter_area;
    
    if (union_area <= 0.0f) return 0.0f;
    return inter_area / union_area;
}

static void nms(float nms_threshold, int num_boxes,
                std::vector<float>& boxes,
                std::vector<int>& classes,
                std::vector<int>& indices,
                int target_class) {
    
    for (int i = 0; i < num_boxes; i++) {
        if (indices[i] == -1 || classes[i] != target_class) continue;
        
        int idx_i = i;
        float x1_min = boxes[idx_i * 4];
        float y1_min = boxes[idx_i * 4 + 1];
        float x1_max = x1_min + boxes[idx_i * 4 + 2];
        float y1_max = y1_min + boxes[idx_i * 4 + 3];
        
        for (int j = i + 1; j < num_boxes; j++) {
            if (indices[j] == -1 || classes[j] != target_class) continue;
            
            int idx_j = j;
            float x2_min = boxes[idx_j * 4];
            float y2_min = boxes[idx_j * 4 + 1];
            float x2_max = x2_min + boxes[idx_j * 4 + 2];
            float y2_max = y2_min + boxes[idx_j * 4 + 3];
            
            float iou = CalculateOverlap(x1_min, y1_min, x1_max, y1_max,
                                         x2_min, y2_min, x2_max, y2_max);
            
            if (iou > nms_threshold) {
                indices[j] = -1;
            }
        }
    }
}

static void quick_sort_indice_inverse(std::vector<float>& scores, int left, int right,
                                       std::vector<int>& indices) {
    if (left >= right) return;
    
    int i = left, j = right;
    float pivot_score = scores[left];
    int pivot_idx = indices[left];
    
    while (i < j) {
        while (i < j && scores[j] <= pivot_score) j--;
        if (i < j) {
            scores[i] = scores[j];
            indices[i] = indices[j];
            i++;
        }
        
        while (i < j && scores[i] >= pivot_score) i++;
        if (i < j) {
            scores[j] = scores[i];
            indices[j] = indices[i];
            j--;
        }
    }
    
    scores[i] = pivot_score;
    indices[i] = pivot_idx;
    
    quick_sort_indice_inverse(scores, left, i - 1, indices);
    quick_sort_indice_inverse(scores, i + 1, right, indices);
}

// ========== 单层处理 ==========

static int process(int8_t* input, const int* anchors,
                   int grid_h, int grid_w,
                   int model_in_w, int model_in_h,
                   int stride,
                   std::vector<float>& boxes,
                   std::vector<float>& scores,
                   std::vector<int>& classes,
                   float conf_threshold,
                   int zero_point, float scale) {
    
    int num_classes = 1;
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    float threshold_unsigmoid = unsigmoid(conf_threshold);
    int8_t threshold_qnt = qnt_f32_to_affine(threshold_unsigmoid, zero_point, scale);
    
    for (int anchor_idx = 0; anchor_idx < 3; anchor_idx++) {
        for (int h = 0; h < grid_h; h++) {
            for (int w = 0; w < grid_w; w++) {
                int base_idx = w + anchor_idx * grid_len * 6 + h * grid_w;
                int8_t* ptr = input + base_idx;
                
                int8_t obj_score_qnt = ptr[(anchor_idx * 6 + 4) * grid_len];
                if (obj_score_qnt <= threshold_qnt) continue;
                
                float box_x = deqnt_affine_to_f32(scale, ptr[0], zero_point);
                box_x = (sigmoid(box_x) * 2.0f - 0.5f);
                
                float box_y = deqnt_affine_to_f32(scale, ptr[grid_len], zero_point);
                box_y = (sigmoid(box_y) * 2.0f - 0.5f);
                
                float box_w = deqnt_affine_to_f32(scale, ptr[grid_len * 2], zero_point);
                box_w = sigmoid(box_w) * 2.0f;
                box_w = box_w * box_w * anchors[anchor_idx * 2];
                
                float box_h = deqnt_affine_to_f32(scale, ptr[grid_len * 3], zero_point);
                box_h = sigmoid(box_h) * 2.0f;
                box_h = box_h * box_h * anchors[anchor_idx * 2 + 1];
                
                box_x = ((float)w + box_x) * stride - box_w / 2.0f;
                box_y = ((float)h + box_y) * stride - box_h / 2.0f;
                
                int8_t class_score_qnt = ptr[grid_len * 5];
                
                if (class_score_qnt > threshold_qnt) {
                    float obj_score = sigmoid(deqnt_affine_to_f32(scale, obj_score_qnt, zero_point));
                    float cls_score = sigmoid(deqnt_affine_to_f32(scale, class_score_qnt, zero_point));
                    float final_score = obj_score * cls_score;
                    
                    boxes.push_back(box_x);
                    boxes.push_back(box_y);
                    boxes.push_back(box_w);
                    boxes.push_back(box_h);
                    scores.push_back(final_score);
                    classes.push_back(0);
                    validCount++;
                }
            }
        }
    }
    
    return validCount;
}

// ========== 主函数（对接postprocess.h接口）==========

extern "C" int person_post_process(int8_t* input0, int8_t* input1, int8_t* input2,
                                    int model_in_w, int model_in_h,
                                    float conf_threshold, float nms_threshold,
                                    std::vector<int>& qnt_zps,
                                    std::vector<float>& qnt_scales,
                                    detect_result_group_t* group) {
    
    memset(group, 0, sizeof(detect_result_group_t));
    
    std::vector<float> boxes;
    std::vector<float> scores;
    std::vector<int> classes;
    
    // 处理三个YOLO层
    int count0 = process(input0, anchor0, model_in_h / 8, model_in_w / 8,
                         model_in_w, model_in_h, 8,
                         boxes, scores, classes,
                         conf_threshold, qnt_zps[0], qnt_scales[0]);
    
    int count1 = process(input1, anchor1, model_in_h / 16, model_in_w / 16,
                         model_in_w, model_in_h, 16,
                         boxes, scores, classes,
                         conf_threshold, qnt_zps[1], qnt_scales[1]);
    
    int count2 = process(input2, anchor2, model_in_h / 32, model_in_w / 32,
                         model_in_w, model_in_h, 32,
                         boxes, scores, classes,
                         conf_threshold, qnt_zps[2], qnt_scales[2]);
    
    int total_count = count0 + count1 + count2;
    
    if (total_count == 0) {
        return 0;
    }
    
    // 按置信度排序
    std::vector<int> indices(total_count);
    for (int i = 0; i < total_count; i++) {
        indices[i] = i;
    }
    quick_sort_indice_inverse(scores, 0, total_count - 1, indices);
    
    // NMS
    std::set<int> class_set(classes.begin(), classes.end());
    for (int cls : class_set) {
        nms(nms_threshold, total_count, boxes, classes, indices, cls);
    }
    
    // 收集结果
    int valid_count = 0;
    for (int i = 0; i < total_count && valid_count < OBJ_NUMB_MAX_SIZE; i++) {
        if (indices[i] == -1) continue;
        
        int idx = indices[i];
        float x1 = boxes[idx * 4];
        float y1 = boxes[idx * 4 + 1];
        float x2 = x1 + boxes[idx * 4 + 2];
        float y2 = y1 + boxes[idx * 4 + 3];
        int cls = classes[idx];
        
        group->results[valid_count].box.left = clamp(x1, 0, model_in_w);
        group->results[valid_count].box.top = clamp(y1, 0, model_in_h);
        group->results[valid_count].box.right = clamp(x2, 0, model_in_w);
        group->results[valid_count].box.bottom = clamp(y2, 0, model_in_h);
        group->results[valid_count].prop = scores[i];
        group->results[valid_count].class_index = cls;
        strncpy(group->results[valid_count].name, labels[cls], OBJ_NAME_MAX_SIZE - 1);
        group->results[valid_count].name[OBJ_NAME_MAX_SIZE - 1] = '\0';
        
        valid_count++;
    }
    
    group->count = valid_count;
    return 0;
}

// 清理函数（兼容性）
void deinitPostProcess() {
    // 无需清理
}
