#include "nafnet_tiny_enhancer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "log.h"
}

namespace {
void* load_model_data(const char* path, int* size) {
    FILE* fp = std::fopen(path, "rb");
    if (!fp) {
        return nullptr;
    }

    std::fseek(fp, 0, SEEK_END);
    long file_size = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    if (file_size <= 0) {
        std::fclose(fp);
        return nullptr;
    }

    void* data = std::malloc(static_cast<size_t>(file_size));
    if (!data) {
        std::fclose(fp);
        return nullptr;
    }

    size_t read_size = std::fread(data, 1, static_cast<size_t>(file_size), fp);
    std::fclose(fp);
    if (read_size != static_cast<size_t>(file_size)) {
        std::free(data);
        return nullptr;
    }

    *size = static_cast<int>(file_size);
    return data;
}

void infer_tensor_shape(const rknn_tensor_attr& attr,
                        int* height,
                        int* width,
                        int* channels,
                        rknn_tensor_format* format) {
    *height = 0;
    *width = 0;
    *channels = 0;
    *format = attr.fmt;

    if (attr.n_dims == 4) {
        if (attr.fmt == RKNN_TENSOR_NCHW) {
            *channels = attr.dims[1];
            *height = attr.dims[2];
            *width = attr.dims[3];
        } else {
            *height = attr.dims[1];
            *width = attr.dims[2];
            *channels = attr.dims[3];
        }
    } else if (attr.n_dims == 3) {
        if (attr.fmt == RKNN_TENSOR_NCHW) {
            *channels = attr.dims[0];
            *height = attr.dims[1];
            *width = attr.dims[2];
        } else {
            *height = attr.dims[0];
            *width = attr.dims[1];
            *channels = attr.dims[2];
        }
    }
}

bool is_float_tensor(rknn_tensor_type type) {
    return type == RKNN_TENSOR_FLOAT32 || type == RKNN_TENSOR_FLOAT16;
}

void pack_nchw_float(const cv::Mat& rgb_float, std::vector<float>* output) {
    const int height = rgb_float.rows;
    const int width = rgb_float.cols;
    output->assign(static_cast<size_t>(height * width * 3), 0.0f);

    const int plane = height * width;
    for (int y = 0; y < height; ++y) {
        const cv::Vec3f* row = rgb_float.ptr<cv::Vec3f>(y);
        for (int x = 0; x < width; ++x) {
            const int idx = y * width + x;
            (*output)[idx] = row[x][0];
            (*output)[plane + idx] = row[x][1];
            (*output)[plane * 2 + idx] = row[x][2];
        }
    }
}

void pack_nchw_u8(const cv::Mat& rgb_u8, std::vector<unsigned char>* output) {
    const int height = rgb_u8.rows;
    const int width = rgb_u8.cols;
    output->assign(static_cast<size_t>(height * width * 3), 0);

    const int plane = height * width;
    for (int y = 0; y < height; ++y) {
        const cv::Vec3b* row = rgb_u8.ptr<cv::Vec3b>(y);
        for (int x = 0; x < width; ++x) {
            const int idx = y * width + x;
            (*output)[idx] = row[x][0];
            (*output)[plane + idx] = row[x][1];
            (*output)[plane * 2 + idx] = row[x][2];
        }
    }
}

void clamp_output_rgb(cv::Mat* image, float upper_bound, float scale) {
    for (int y = 0; y < image->rows; ++y) {
        cv::Vec3f* row = image->ptr<cv::Vec3f>(y);
        for (int x = 0; x < image->cols; ++x) {
            for (int c = 0; c < 3; ++c) {
                float value = row[x][c];
                if (!std::isfinite(value)) {
                    value = 0.0f;
                }
                value = std::max(0.0f, std::min(upper_bound, value));
                row[x][c] = value * scale;
            }
        }
    }
}
}

NAFNetTinyEnhancer::NAFNetTinyEnhancer(const std::string& modelPath)
    : modelPath(modelPath) {}

NAFNetTinyEnhancer::~NAFNetTinyEnhancer() {
    release();
}

bool NAFNetTinyEnhancer::isReady() {
    if (!initTried) {
        init();
    }
    return ready;
}

bool NAFNetTinyEnhancer::init() {
    initTried = true;
    if (ready) {
        return true;
    }

    int model_size = 0;
    void* model_data = load_model_data(modelPath.c_str(), &model_size);
    if (!model_data) {
        log_warn("NAFNetTinyEnhancer: model file not found or unreadable: %s", modelPath.c_str());
        return false;
    }

    int ret = rknn_init(&ctx, model_data, model_size, 0, nullptr);
    std::free(model_data);
    if (ret < 0) {
        ctx = 0;
        log_error("NAFNetTinyEnhancer: rknn_init failed ret=%d path=%s", ret, modelPath.c_str());
        return false;
    }

    rknn_input_output_num io_num;
    std::memset(&io_num, 0, sizeof(io_num));
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0 || io_num.n_input < 1 || io_num.n_output < 1) {
        log_error("NAFNetTinyEnhancer: query io num failed ret=%d", ret);
        release();
        return false;
    }

    rknn_tensor_attr input_attr;
    std::memset(&input_attr, 0, sizeof(input_attr));
    input_attr.index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr));
    if (ret < 0) {
        log_error("NAFNetTinyEnhancer: query input attr failed ret=%d", ret);
        release();
        return false;
    }

    infer_tensor_shape(input_attr, &inputHeight, &inputWidth, &inputChannels, &inputFmt);
    inputType = input_attr.type;
    if (inputWidth <= 0 || inputHeight <= 0 || inputChannels != 3) {
        log_error("NAFNetTinyEnhancer: unsupported input shape h=%d w=%d c=%d", inputHeight, inputWidth, inputChannels);
        release();
        return false;
    }

    rknn_tensor_attr output_attr;
    std::memset(&output_attr, 0, sizeof(output_attr));
    output_attr.index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attr, sizeof(output_attr));
    if (ret < 0) {
        log_error("NAFNetTinyEnhancer: query output attr failed ret=%d", ret);
        release();
        return false;
    }
    outputFmt = output_attr.fmt;

    ready = true;
    log_info(
        "NAFNetTinyEnhancer: loaded %s input=%dx%dx%d input_fmt=%d input_type=%d output_fmt=%d",
        modelPath.c_str(),
        inputWidth,
        inputHeight,
        inputChannels,
        static_cast<int>(inputFmt),
        static_cast<int>(inputType),
        static_cast<int>(outputFmt));
    return true;
}

void NAFNetTinyEnhancer::release() {
    if (ctx) {
        rknn_destroy(ctx);
        ctx = 0;
    }
    ready = false;
}

cv::Mat NAFNetTinyEnhancer::enhance(const cv::Mat& input) {
    if (input.empty() || input.cols <= 0 || input.rows <= 0) {
        return cv::Mat();
    }
    if (!isReady()) {
        return cv::Mat();
    }

    cv::Mat resized;
    int interpolation = (input.cols < inputWidth || input.rows < inputHeight) ? cv::INTER_CUBIC : cv::INTER_AREA;
    cv::resize(input, resized, cv::Size(inputWidth, inputHeight), 0, 0, interpolation);

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    std::vector<float> float_input_buffer;
    std::vector<unsigned char> u8_input_buffer;
    void* input_buf = nullptr;
    size_t input_size = 0;
    rknn_tensor_type feed_type = inputType;
    rknn_tensor_format feed_fmt = inputFmt == RKNN_TENSOR_UNDEFINED ? RKNN_TENSOR_NHWC : inputFmt;

    if (is_float_tensor(inputType)) {
        cv::Mat rgb_float;
        rgb.convertTo(rgb_float, CV_32FC3, 1.0 / 255.0);

        if (feed_fmt == RKNN_TENSOR_NCHW) {
            pack_nchw_float(rgb_float, &float_input_buffer);
            input_buf = float_input_buffer.data();
        } else {
            if (!rgb_float.isContinuous()) {
                rgb_float = rgb_float.clone();
            }
            input_buf = rgb_float.data;
        }

        input_size = static_cast<size_t>(inputWidth * inputHeight * inputChannels * sizeof(float));
        feed_type = RKNN_TENSOR_FLOAT32;
    } else {
        if (feed_fmt == RKNN_TENSOR_NCHW) {
            pack_nchw_u8(rgb, &u8_input_buffer);
            input_buf = u8_input_buffer.data();
        } else {
            input_buf = rgb.data;
        }

        input_size = static_cast<size_t>(inputWidth * inputHeight * inputChannels);
        feed_type = RKNN_TENSOR_UINT8;
    }

    rknn_input inputs[1];
    std::memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = feed_type;
    inputs[0].size = input_size;
    inputs[0].fmt = feed_fmt;
    inputs[0].buf = input_buf;
    inputs[0].pass_through = 0;

    int ret = rknn_inputs_set(ctx, 1, inputs);
    if (ret < 0) {
        log_error("NAFNetTinyEnhancer: rknn_inputs_set failed ret=%d", ret);
        return cv::Mat();
    }

    ret = rknn_run(ctx, nullptr);
    if (ret < 0) {
        log_error("NAFNetTinyEnhancer: rknn_run failed ret=%d", ret);
        return cv::Mat();
    }

    rknn_output outputs[1];
    std::memset(outputs, 0, sizeof(outputs));
    outputs[0].index = 0;
    outputs[0].want_float = 1;
    ret = rknn_outputs_get(ctx, 1, outputs, nullptr);
    if (ret < 0 || !outputs[0].buf) {
        log_error("NAFNetTinyEnhancer: rknn_outputs_get failed ret=%d", ret);
        return cv::Mat();
    }

    rknn_tensor_attr output_attr;
    std::memset(&output_attr, 0, sizeof(output_attr));
    output_attr.index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attr, sizeof(output_attr));
    int out_h = inputHeight;
    int out_w = inputWidth;
    int out_c = 3;
    if (ret == 0) {
        infer_tensor_shape(output_attr, &out_h, &out_w, &out_c, &outputFmt);
    }

    cv::Mat output_rgb(out_h, out_w, CV_32FC3);
    const float* out_ptr = static_cast<const float*>(outputs[0].buf);
    double max_val = -1e12;
    int plane = out_h * out_w;

    if (outputFmt == RKNN_TENSOR_NCHW) {
        for (int y = 0; y < out_h; ++y) {
            cv::Vec3f* row = output_rgb.ptr<cv::Vec3f>(y);
            for (int x = 0; x < out_w; ++x) {
                int idx = y * out_w + x;
                row[x][0] = out_ptr[idx];
                row[x][1] = out_ptr[plane + idx];
                row[x][2] = out_ptr[plane * 2 + idx];
                max_val = std::max(max_val, static_cast<double>(std::max(row[x][0], std::max(row[x][1], row[x][2]))));
            }
        }
    } else {
        for (int y = 0; y < out_h; ++y) {
            cv::Vec3f* row = output_rgb.ptr<cv::Vec3f>(y);
            for (int x = 0; x < out_w; ++x) {
                int idx = (y * out_w + x) * 3;
                row[x][0] = out_ptr[idx];
                row[x][1] = out_ptr[idx + 1];
                row[x][2] = out_ptr[idx + 2];
                max_val = std::max(max_val, static_cast<double>(std::max(row[x][0], std::max(row[x][1], row[x][2]))));
            }
        }
    }

    rknn_outputs_release(ctx, 1, outputs);

    if (max_val <= 1.5) {
        clamp_output_rgb(&output_rgb, 1.0f, 255.0f);
    } else {
        clamp_output_rgb(&output_rgb, 255.0f, 1.0f);
    }

    cv::Mat output_u8;
    output_rgb.convertTo(output_u8, CV_8UC3);
    cv::Mat output_bgr;
    cv::cvtColor(output_u8, output_bgr, cv::COLOR_RGB2BGR);

    cv::Mat restored;
    cv::resize(output_bgr, restored, input.size(), 0, 0, cv::INTER_CUBIC);
    return restored;
}
