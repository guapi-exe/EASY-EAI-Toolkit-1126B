/**
 * decode.cpp - 人脸框和关键点解码（重写版）
 * 
 * 基于反编译代码重写，将模型输出解码为人脸检测结果
 */

#include "decode.h"
#include "generator.h"
#include "tools.h"
#include "face_detect.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <sys/time.h>

/**
 * 获取当前时间戳（辅助函数）
 * @return 当前时间的秒数（包含微秒）
 */
extern "C" double what_time_is_it_now() {
    struct timeval tv;
    if (gettimeofday(&tv, NULL) == 0) {
        return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
    }
    return 0.0;
}

/**
 * 解码人脸框和关键点
 * 
 * @param confident 置信度数组，每个prior有2个值[bg_score, face_score]
 * @param loc 位置偏移数组，每个prior有4个值[dx, dy, dw, dh]
 * @param predict 关键点偏移数组，每个prior有10个值[x0,y0, x1,y1, ..., x4,y4]
 * @param score_threshold 置信度阈值
 * @param nms_threshold NMS IoU阈值
 * @param prior_data 先验框数据
 * @param outputs 输出的检测结果
 */
void decode_box_and_landmark(float* confident, 
                             float* loc, 
                             float* predict, 
                             float score_threshold, 
                             float nms_threshold,
                             std::vector<std::vector<float>>& prior_data,
                             std::vector<det>& outputs) {
    int num_priors = prior_data.size();
    std::vector<det> temp_results;
    
    // 遍历所有先验框
    for (int i = 0; i < num_priors; i++) {
        // 获取face类别的置信度（索引1）
        float face_score = confident[i * 2 + 1];
        
        // 如果置信度超过阈值
        if (face_score > score_threshold) {
            det detection;
            
            // 解码人脸框
            float* box_offset = loc + (i * 4);  // 每个prior有4个位置偏移值
            decode_box(box_offset, prior_data[i], detection.box);
            
            // 解码人脸关键点
            float* landmark_offset = predict + (i * 10);  // 每个prior有10个关键点值
            decode_landmark(landmark_offset, prior_data[i], detection.landmarks);
            
            // 设置置信度
            detection.score = face_score;
            
            temp_results.push_back(detection);
        }
    }
    
    // 应用NMS非极大值抑制
    nms_cpu(temp_results, nms_threshold, outputs);
}
