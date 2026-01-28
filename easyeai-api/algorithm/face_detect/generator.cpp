/**
 * generator.cpp - Prior数据生成和解码模块（重写版）
 * 
 * 基于反编译代码重写，保持与原始库相同的算法逻辑
 */

#include "generator.h"
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>

// 全局配置（根据反编译代码推测）
static const int min_sizes[][2] = {{16, 32}, {64, 128}, {256, 512}};
static const int steps[] = {8, 16, 32};
static const int num_steps = 3;
static const int num_anchors_per_loc = 2;
static const float variance[2] = {0.1f, 0.2f};

/**
 * 生成prior数据
 * 
 * @param width 图像宽度
 * @param height 图像高度
 * @return 二维vector，每个元素包含4个值 [cx, cy, w, h]
 */
std::vector<std::vector<float>> generate_prior_data(int width, int height) {
    std::vector<std::vector<float>> priors;
    
    int image_shape[2] = {height, width};
    int feature_map_dims[3][2];
    
    // 计算每个特征层的尺寸
    for (int i = 0; i < num_steps; i++) {
        for (int j = 0; j < 2; j++) {
            feature_map_dims[i][j] = (int)std::ceil((float)image_shape[j] / (float)steps[i]);
        }
    }
    
    // 为每个特征层生成anchors
    for (int k = 0; k < num_steps; k++) {
        int f_h = feature_map_dims[k][0];
        int f_w = feature_map_dims[k][1];
        
        for (int i = 0; i < f_h; i++) {
            for (int j = 0; j < f_w; j++) {
                for (int anchor_idx = 0; anchor_idx < num_anchors_per_loc; anchor_idx++) {
                    std::vector<float> anchor(4);
                    
                    // 计算anchor的宽高
                    int min_size = min_sizes[k][anchor_idx];
                    anchor[2] = (float)min_size / (float)width;   // w
                    anchor[3] = (float)min_size / (float)height;  // h
                    
                    // 计算anchor的中心点
                    anchor[0] = ((float)j + 0.5f) * (float)steps[k] / (float)width;  // cx
                    anchor[1] = ((float)i + 0.5f) * (float)steps[k] / (float)height; // cy
                    
                    priors.push_back(anchor);
                }
            }
        }
    }
    
    return priors;
}

/**
 * 解码单个人脸框
 * 
 * @param loc 模型预测的位置偏移 [dx, dy, dw, dh]
 * @param single_prior_data 单个prior数据 [cx, cy, w, h]
 * @param rect 输出的矩形框
 */
void decode_box(float* loc, const std::vector<float>& single_prior_data, cv::Rect_<float>& rect) {
    std::vector<float> box(4);
    
    // 解码中心点
    box[0] = single_prior_data[0] + loc[0] * variance[0] * single_prior_data[2];
    box[1] = single_prior_data[1] + loc[1] * variance[0] * single_prior_data[3];
    
    // 解码宽高
    box[2] = single_prior_data[2] * std::exp(loc[2] * variance[1]);
    box[3] = single_prior_data[3] * std::exp(loc[3] * variance[1]);
    
    // 转换为 [x, y, w, h] 格式（左上角坐标）
    box[0] -= box[2] / 2.0f;
    box[1] -= box[3] / 2.0f;
    
    rect.x = box[0];
    rect.y = box[1];
    rect.width = box[2];
    rect.height = box[3];
}

/**
 * 解码人脸关键点
 * 
 * @param predict 模型预测的关键点偏移 [x0, y0, x1, y1, ..., x4, y4]
 * @param single_prior_data 单个prior数据 [cx, cy, w, h]
 * @param landmark 输出的5个关键点
 */
void decode_landmark(float* predict, const std::vector<float>& single_prior_data, 
                     std::vector<cv::Point2f>& landmark) {
    if (landmark.size() != 5) {
        printf("error, landmark size is %zu, instead of 5\n", landmark.size());
        return;
    }
    
    for (int i = 0; i < 5; i++) {
        landmark[i].x = single_prior_data[0] + predict[i * 2] * variance[0] * single_prior_data[2];
        landmark[i].y = single_prior_data[1] + predict[i * 2 + 1] * variance[0] * single_prior_data[3];
    }
}
