/**
 * tools.cpp - 人脸检测辅助工具（重写版）
 * 
 * 基于反编译代码重写，包含NMS和letter_box预处理
 */

#include "face_detect.h"
#include "tools.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <vector>

/**
 * NMS非极大值抑制
 * 
 * @param boxes 待处理的检测框
 * @param threshold IoU阈值，超过此值的框会被抑制
 * @param filtered_output 过滤后的输出结果
 */
void nms_cpu(std::vector<det>& boxes, float threshold, std::vector<det>& filtered_output) {
    filtered_output.clear();
    
    if (boxes.empty()) {
        return;
    }
    
    // 创建索引数组
    std::vector<int> indices(boxes.size());
    for (size_t i = 0; i < boxes.size(); i++) {
        indices[i] = i;
    }
    
    // 按置信度降序排序
    std::sort(indices.begin(), indices.end(), [&boxes](int i1, int i2) {
        return boxes[i1].score > boxes[i2].score;
    });
    
    // NMS主循环
    while (!indices.empty()) {
        int current_idx = indices[0];
        filtered_output.push_back(boxes[current_idx]);
        
        std::vector<int> remaining;
        for (size_t i = 1; i < indices.size(); i++) {
            int compare_idx = indices[i];
            
            // 计算两个框的交集
            float x1 = std::max(boxes[current_idx].box.x, boxes[compare_idx].box.x);
            float y1 = std::max(boxes[current_idx].box.y, boxes[compare_idx].box.y);
            
            float x2 = std::min(boxes[current_idx].box.x + boxes[current_idx].box.width,
                               boxes[compare_idx].box.x + boxes[compare_idx].box.width);
            float y2 = std::min(boxes[current_idx].box.y + boxes[current_idx].box.height,
                               boxes[compare_idx].box.y + boxes[compare_idx].box.height);
            
            float inter_width = std::max(0.0f, x2 - x1);
            float inter_height = std::max(0.0f, y2 - y1);
            float inter_area = inter_width * inter_height;
            
            // 计算两个框的面积
            float area1 = boxes[current_idx].box.width * boxes[current_idx].box.height;
            float area2 = boxes[compare_idx].box.width * boxes[compare_idx].box.height;
            
            // 计算IoU
            float iou = inter_area / (area1 + area2 - inter_area);
            
            // 如果IoU小于阈值，保留该框
            if (iou <= threshold) {
                remaining.push_back(compare_idx);
            }
        }
        
        indices = remaining;
    }
}

/**
 * letter_box图像预处理
 * 等比例缩放图像到目标尺寸，不足部分用灰色填充
 * 
 * @param src 输入图像
 * @param dst 输出图像
 * @param width 目标宽度
 * @param height 目标高度
 * @param t_info 记录变换信息的结构体
 */
void letter_box(cv::Mat& src, cv::Mat& dst, int width, int height, Transform_info* t_info) {
    int src_width = src.cols;
    int src_height = src.rows;
    
    // 计算缩放比例
    float target_ratio = (float)width / (float)height;
    float src_ratio = (float)src_width / (float)src_height;
    
    int pad_left, pad_right, pad_top, pad_bottom;
    
    // 根据宽高比决定填充方向
    if (src_ratio < target_ratio) {
        // 源图像更窄，需要在左右填充
        int pad_width = (int)((float)src_height * target_ratio - (float)src_width);
        pad_left = pad_width / 2;
        pad_right = pad_width - pad_left;
        pad_top = 0;
        pad_bottom = 0;
    } else {
        // 源图像更宽，需要在上下填充
        int pad_height = (int)((float)src_width / target_ratio - (float)src_height);
        pad_top = pad_height / 2;
        pad_bottom = pad_height - pad_top;
        pad_left = 0;
        pad_right = 0;
    }
    
    // 填充边界（灰色114）
    cv::Mat padded;
    cv::copyMakeBorder(src, padded, 
                       pad_top, pad_bottom, 
                       pad_left, pad_right,
                       cv::BORDER_CONSTANT, 
                       cv::Scalar(114, 114, 114));
    
    // 记录变换信息
    if (t_info != nullptr) {
        t_info->src_width = src_width;
        t_info->src_height = src_height;
        t_info->target_width = width;
        t_info->target_height = height;
        t_info->top = pad_top;
        t_info->bottom = pad_bottom;
        t_info->left = pad_left;
        t_info->right = pad_right;
        t_info->ratio = (float)padded.cols / (float)width;
    }
    
    // 缩放到目标尺寸（OpenCV 4.6兼容写法）
    cv::resize(padded, dst, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
}
