#include "send_data.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <ctime>
#include "base64.h"
#include "cJSON.h"

std::string image_to_base64(const cv::Mat& img, const std::string& ext = ".jpg") {
    std::vector<uchar> buf;
    if (!cv::imencode(ext, img, buf)) {
        return "";  
    }

    size_t out_buf_size = ((buf.size() + 2) / 3) * 4 + 4;
    std::vector<char> base64_buf(out_buf_size, 0);

    int32_t encoded_len = base64_encode(
        base64_buf.data(),
        reinterpret_cast<const char*>(buf.data()),
        static_cast<unsigned int>(buf.size())
    );

    if (encoded_len <= 0) {
        return "";
    }

    return std::string(base64_buf.data(), encoded_len);
}

std::string get_current_time_string() {
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    return std::string(buf);
}

std::string build_json(const cv::Mat& img, int id, const std::string& type) {
    std::string base64_str = image_to_base64(img);
    if (base64_str.empty()) {
        std::cerr << "Failed to convert image to base64\n";
        return "";
    }

    cJSON* root = cJSON_CreateObject();
    cJSON_AddNumberToObject(root, "id", id);
    cJSON_AddStringToObject(root, "type", type.c_str());
    cJSON_AddStringToObject(root, "time", get_current_time_string().c_str());
    cJSON_AddStringToObject(root, "image", base64_str.c_str());

    char* json_str = cJSON_PrintUnformatted(root);  
    std::string result(json_str);

    cJSON_free(json_str);
    cJSON_Delete(root);

    return result;
}