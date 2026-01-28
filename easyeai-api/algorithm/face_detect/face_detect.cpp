/**
 * face_detect.cpp - 人脸检测核心模块（重构版）
 * 
 * 移除硬件校验，修复OpenCV 4.6兼容性
 */

#include "face_detect.h"
#include "tools.h"
#include "generator.h"
#include "decode.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cstdlib>

// 全局prior数据（正确类型）
static std::vector<std::vector<float>> prior_data;
static bool prior_data_initialized = false;

// 声明外部函数
extern "C" {
    int decrypte_init(uint16_t param1, int param2);
    int decrypte_model(const void* src, void* dest, int size);
}

// 加载模型文件
static void* load_model(const char* path, int* size) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        printf("fopen %s fail!\n", path);
        return nullptr;
    }
    
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    void* data = malloc(file_size);
    if (!data) {
        fclose(fp);
        return nullptr;
    }
    
    size_t read_size = fread(data, 1, file_size, fp);
    fclose(fp);
    
    if (read_size != (size_t)file_size) {
        printf("fread %s fail!\n", path);
        free(data);
        return nullptr;
    }
    
    *size = (int)file_size;
    return data;
}

// 解密模型数据
static void* decrypt_model_data(void* encrypted_data, int encrypted_size, int* decrypted_size) {
    void* decrypted_data = malloc(encrypted_size);
    if (!decrypted_data) {
        return nullptr;
    }
    
    int ret = decrypte_model(encrypted_data, decrypted_data, encrypted_size);
    if (ret != 0) {
        free(decrypted_data);
        return nullptr;
    }
    
    *decrypted_size = encrypted_size - 4;
    return decrypted_data;
}

// 公共API实现

int face_detect_init(rknn_context* ctx, const char* model_path) {
    if (!ctx || !model_path) {
        return -1;
    }
    
    // 初始化解密模块
    decrypte_init(1, 0);
    
    // 加载加密模型
    int encrypted_size = 0;
    void* encrypted_data = load_model(model_path, &encrypted_size);
    
    if (!encrypted_data) {
        return -1;
    }
    
    // 解密模型
    int decrypted_size = 0;
    void* decrypted_data = decrypt_model_data(encrypted_data, encrypted_size, &decrypted_size);
    
    free(encrypted_data);
    
    if (!decrypted_data) {
        printf("Failed to decrypt model\n");
        return -1;
    }
    
    // 初始化RKNN
    int ret = rknn_init(ctx, decrypted_data, decrypted_size, 0, nullptr);
    
    free(decrypted_data);
    
    if (ret < 0) {
        printf("model init fail! ret=%d\n", ret);
        return -1;
    }
    
    // 查询模型信息
    rknn_input_output_num io_num;
    ret = rknn_query(*ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret == 0) {
        printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);
    } else {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    
    return 0;
}

int face_detect_release(rknn_context ctx) {
    rknn_destroy(ctx);
    return 0;
}

int face_detect_run(rknn_context ctx, cv::Mat& img, std::vector<det>& dets) {
    if (img.empty()) {
        return -1;
    }
    
    // Letter box预处理
    cv::Mat letterboxed;
    Transform_info transform_info;
    letter_box(img, letterboxed, 300, 300, &transform_info);
    
    // 初始化prior数据（只初始化一次）
    if (!prior_data_initialized) {
        prior_data = generate_prior_data(300, 300);
        prior_data_initialized = true;
        printf("Prior data initialized: %zu priors\n", prior_data.size());
    }
    
    // 设置输入
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = 300 * 300 * 3;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = letterboxed.data;
    
    int ret = rknn_inputs_set(ctx, 1, inputs);
    if (ret < 0) {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }
    
    // 执行推理
    ret = rknn_run(ctx, nullptr);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }
    
    // 获取输出
    rknn_output outputs[3];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < 3; i++) {
        outputs[i].index = i;
        outputs[i].want_float = 1;
    }
    
    ret = rknn_outputs_get(ctx, 3, outputs, nullptr);
    if (ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return -1;
    }
    
    // 解码检测结果（参数顺序：confident, loc, predict）
    float* confident = (float*)outputs[1].buf;  // 置信度
    float* loc = (float*)outputs[0].buf;        // 位置
    float* predict = (float*)outputs[2].buf;    // 关键点
    
    decode_box_and_landmark(confident, loc, predict, 0.4f, 0.4f, prior_data, dets);
    
    // 坐标变换回原图（检测结果是归一化坐标[0,1]，需要先乘以300，再减去padding，最后除以ratio恢复原图尺寸）
    for (size_t i = 0; i < dets.size(); i++) {
        dets[i].box.x = (dets[i].box.x * 300.0f - transform_info.left) * transform_info.ratio;
        dets[i].box.y = (dets[i].box.y * 300.0f - transform_info.top) * transform_info.ratio;
        dets[i].box.width = dets[i].box.width * 300.0f * transform_info.ratio;
        dets[i].box.height = dets[i].box.height * 300.0f * transform_info.ratio;
        
        for (size_t j = 0; j < dets[i].landmarks.size(); j++) {
            dets[i].landmarks[j].x = (dets[i].landmarks[j].x * 300.0f - transform_info.left) * transform_info.ratio;
            dets[i].landmarks[j].y = (dets[i].landmarks[j].y * 300.0f - transform_info.top) * transform_info.ratio;
        }
    }
    
    // 释放输出
    rknn_outputs_release(ctx, 3, outputs);
    
    return (int)dets.size();
}
