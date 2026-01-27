/**
 * OpenCV 兼容层
 * 
 * RKNN 库中的 letter_box 函数使用了与 OpenCV 4.6 不兼容的调用方式
 * 这个文件提供了兼容的 letter_box 实现
 */

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>

extern "C" {

/**
 * Letter box 图像预处理 - 保持宽高比缩放并填充
 * 
 * @param src 源图像指针
 * @param dst 目标图像指针
 * @param target_size 目标尺寸（正方形）
 * @return 0 表示成功
 */
int letter_box(cv::Mat* src, cv::Mat* dst, int target_size) {
    if (src == nullptr || dst == nullptr || src->empty() || target_size <= 0) {
        return -1;
    }
    
    int src_h = src->rows;
    int src_w = src->cols;
    
    if (src_h <= 0 || src_w <= 0) {
        return -1;
    }
    
    // 计算缩放比例（保持宽高比）
    float scale = std::min((float)target_size / (float)src_w, 
                           (float)target_size / (float)src_h);
    
    if (scale <= 0) {
        return -1;
    }
    
    // 计算缩放后的尺寸
    int new_w = (int)std::round((float)src_w * scale);
    int new_h = (int)std::round((float)src_h * scale);
    
    // 确保尺寸有效
    if (new_w <= 0) new_w = 1;
    if (new_h <= 0) new_h = 1;
    if (new_w > target_size) new_w = target_size;
    if (new_h > target_size) new_h = target_size;
    
    // 计算填充大小
    int pad_w = target_size - new_w;
    int pad_h = target_size - new_h;
    
    int top = pad_h / 2;
    int bottom = pad_h - top;
    int left = pad_w / 2;
    int right = pad_w - left;
    
    // 缩放图像
    cv::Mat resized;
    cv::resize(*src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    
    // 添加边框填充（灰色填充值：114）
    cv::copyMakeBorder(resized, *dst, top, bottom, left, right, 
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    
    return 0;
}

} // extern "C"
