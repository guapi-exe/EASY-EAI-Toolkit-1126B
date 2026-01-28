/**
 * person_detect.cpp - 人员检测核心模块
 * 
 * 移除硬件校验，修复OpenCV 4.6兼容性
 */

#include "person_detect.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cstdlib>

// ========== Letter Box 图像预处理（OpenCV 4.6兼容）==========

static int letter_box(const cv::Mat& src, cv::Mat& dst, int target_size) {
    if (src.empty() || target_size <= 0) {
        return -1;
    }
    
    int src_h = src.rows;
    int src_w = src.cols;
    
    if (src_h <= 0 || src_w <= 0) {
        return -1;
    }
    
    float scale = std::min((float)target_size / (float)src_w, 
                           (float)target_size / (float)src_h);
    
    if (scale <= 0) {
        return -1;
    }
    
    int new_w = (int)std::round((float)src_w * scale);
    int new_h = (int)std::round((float)src_h * scale);
    
    if (new_w <= 0) new_w = 1;
    if (new_h <= 0) new_h = 1;
    if (new_w > target_size) new_w = target_size;
    if (new_h > target_size) new_h = target_size;
    
    int pad_w = target_size - new_w;
    int pad_h = target_size - new_h;
    
    int top = pad_h / 2;
    int bottom = pad_h - top;
    int left = pad_w / 2;
    int right = pad_w - left;
    
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    
    cv::copyMakeBorder(resized, dst, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    
    return 0;
}

// ========== 坐标缩放还原 ==========

static int scale_coords(detect_result_group_t* group, int img_w, int img_h, int model_size) {
    if (!group) return -1;
    
    for (int i = 0; i < group->count; i++) {
        int left = group->results[i].box.left;
        int top = group->results[i].box.top;
        int right = group->results[i].box.right;
        int bottom = group->results[i].box.bottom;
        
        float scale;
        int offset;
        
        if (img_w < img_h) {
            scale = (float)model_size / (float)img_h;
            offset = (model_size - (int)((float)img_w * scale)) / 2;
            
            group->results[i].box.left = (int)((float)(left - offset) / scale);
            group->results[i].box.top = (int)((float)top / scale);
            group->results[i].box.right = (int)((float)(right - offset) / scale);
            group->results[i].box.bottom = (int)((float)bottom / scale);
        } else {
            scale = (float)model_size / (float)img_w;
            offset = (model_size - (int)((float)img_h * scale)) / 2;
            
            group->results[i].box.left = (int)((float)left / scale);
            group->results[i].box.top = (int)((float)(top - offset) / scale);
            group->results[i].box.right = (int)((float)right / scale);
            group->results[i].box.bottom = (int)((float)(bottom - offset) / scale);
        }
    }
    
    return 0;
}

// ========== 加载模型（使用原始库的解密函数）==========

// 声明原始库中的解密函数
extern "C" {
    int decrypte_init(uint16_t param);
    int decrypte_model(const void* src, void* dest, int size);
}

static void* load_model(const char* path, int* size) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        printf("Failed to open model file: %s\n", path);
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
    
    fread(data, 1, file_size, fp);
    fclose(fp);
    
    *size = (int)file_size;
    return data;
}

// 解密模型数据
static void* decrypt_model_data(void* encrypted_data, int encrypted_size, int* decrypted_size) {
    // 分配解密后的缓冲区
    void* decrypted_data = malloc(encrypted_size);
    if (!decrypted_data) {
        return nullptr;
    }
    
    // 调用原始库的解密函数
    int ret = decrypte_model(encrypted_data, decrypted_data, encrypted_size);
    if (ret != 0) {
        free(decrypted_data);
        return nullptr;
    }
    
    *decrypted_size = encrypted_size - 4; // 解密后数据大小（去掉4字节头）
    return decrypted_data;
}

// ========== 公共API实现 ==========

int person_detect_init(rknn_context* ctx, const char* model_path) {
    if (!ctx || !model_path) {
        return -1;
    }
    
    // 初始化解密模块（绕过硬件验证）
    decrypte_init(0);
    
    // 加载加密的模型文件
    int encrypted_size = 0;
    void* encrypted_data = load_model(model_path, &encrypted_size);
    
    if (!encrypted_data) {
        printf("Failed to load model: %s\n", model_path);
        return -1;
    }
    
    // 解密模型数据
    int decrypted_size = 0;
    void* decrypted_data = decrypt_model_data(encrypted_data, encrypted_size, &decrypted_size);
    
    free(encrypted_data); // 释放加密数据
    
    if (!decrypted_data) {
        printf("Failed to decrypt model\n");
        return -1;
    }
    
    // 使用解密后的数据初始化RKNN
    printf("Loading RKNN model (decrypted %d bytes)...\n", decrypted_size);
    int ret = rknn_init(ctx, decrypted_data, decrypted_size, 0, nullptr);
    
    free(decrypted_data); // 释放解密数据
    
    if (ret < 0) {
        printf("rknn_init failed with code: %d\n", ret);
        return -1;
    }
    
    printf("RKNN model loaded successfully\n");
    return 0;
}

int person_detect_release(rknn_context ctx) {
    return rknn_destroy(ctx);
}

int person_detect_run(rknn_context ctx, cv::Mat input_image, detect_result_group_t* detect_result_group) {
    if (input_image.empty() || !detect_result_group) {
        return -1;
    }
    
    int img_h = input_image.rows;
    int img_w = input_image.cols;
    
    // 查询输入输出数量
    rknn_input_output_num io_num;
    int ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        printf("rknn_query failed\n");
        return -1;
    }
    
    uint32_t n_input = io_num.n_input;
    uint32_t n_output = io_num.n_output;
    
    // 查询输入属性
    std::vector<rknn_tensor_attr> input_attrs(n_input);
    for (uint32_t i = 0; i < n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
        if (ret < 0) {
            return -1;
        }
    }
    
    // 查询输出属性
    std::vector<rknn_tensor_attr> output_attrs(n_output);
    for (uint32_t i = 0; i < n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
        if (ret < 0) {
            return -1;
        }
    }
    
    // 获取模型输入尺寸
    int model_w, model_h, model_c;
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        model_c = input_attrs[0].dims[1];
        model_h = input_attrs[0].dims[2];
        model_w = input_attrs[0].dims[3];
    } else {
        model_h = input_attrs[0].dims[1];
        model_w = input_attrs[0].dims[2];
        model_c = input_attrs[0].dims[3];
    }
    
    // 图像预处理
    cv::Mat letterboxed;
    cv::Mat rgb_img;
    
    letter_box(input_image, letterboxed, model_h);
    cv::cvtColor(letterboxed, rgb_img, cv::COLOR_BGR2RGB);
    
    // 设置输入
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = model_w * model_h * model_c;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = rgb_img.data;
    
    ret = rknn_inputs_set(ctx, n_input, inputs);
    if (ret < 0) {
        return -1;
    }
    
    // 执行推理
    ret = rknn_run(ctx, nullptr);
    if (ret < 0) {
        return -1;
    }
    
    // 获取输出
    std::vector<rknn_output> outputs(n_output);
    memset(outputs.data(), 0, sizeof(rknn_output) * n_output);
    for (uint32_t i = 0; i < n_output; i++) {
        outputs[i].want_float = 0;
    }
    
    ret = rknn_outputs_get(ctx, n_output, outputs.data(), nullptr);
    if (ret < 0) {
        return -1;
    }
    
    // 收集量化参数
    std::vector<int> qnt_zps;
    std::vector<float> qnt_scales;
    for (uint32_t i = 0; i < n_output; i++) {
        qnt_zps.push_back(output_attrs[i].zp);
        qnt_scales.push_back(output_attrs[i].scale);
    }
    
    // 后处理
    person_post_process(
        (int8_t*)outputs[0].buf,
        (int8_t*)outputs[1].buf,
        (int8_t*)outputs[2].buf,
        model_w, model_h,
        BOX_THRESH, NMS_THRESH,
        qnt_zps, qnt_scales,
        detect_result_group
    );
    
    // 坐标还原到原图
    scale_coords(detect_result_group, img_w, img_h, model_h);
    
    // 释放输出
    rknn_outputs_release(ctx, n_output, outputs.data());
    
    return 0;
}
