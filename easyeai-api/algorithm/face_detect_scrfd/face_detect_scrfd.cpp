#include "face_detect_scrfd.h"
#include <rknn_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <math.h>
#include <string.h>

static SCRFDConfig g_config;
static std::vector<std::vector<cv::Point2f>> g_anchor_centers;

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* ===================== Anchor ===================== */
static void generate_anchors(const SCRFDConfig& cfg,
                             std::vector<std::vector<cv::Point2f>>& anchors) {
    anchors.clear();
    anchors.resize(cfg.strides.size());

    for (size_t i = 0; i < cfg.strides.size(); ++i) {
        int stride = cfg.strides[i];
        int fh = cfg.input_height / stride;
        int fw = cfg.input_width / stride;

        for (int y = 0; y < fh; ++y) {
            for (int x = 0; x < fw; ++x) {
                float cx = (x + 0.5f) * stride;
                float cy = (y + 0.5f) * stride;

                /* SCRFD-1G: 每点 2 anchors */
                anchors[i].push_back({cx, cy});
                anchors[i].push_back({cx, cy});
            }
        }
    }
}

/* ===================== Config ===================== */
SCRFDConfig get_scrfd_config(int h, int w) {
    SCRFDConfig cfg;
    cfg.input_height = h;
    cfg.input_width  = w;
    cfg.strides = {8, 16, 32};
    cfg.conf_thresh = 0.45f;
    cfg.nms_thresh  = 0.4f;
    return cfg;
}

/* ===================== Init ===================== */
int face_detect_scrfd_init(rknn_context* ctx,
                           const char* model_path,
                           SCRFDConfig* config) {
    g_config = *config;
    generate_anchors(g_config, g_anchor_centers);

    FILE* fp = fopen(model_path, "rb");
    fseek(fp, 0, SEEK_END);
    int len = ftell(fp);
    rewind(fp);

    void* model = malloc(len);
    fread(model, 1, len, fp);
    fclose(fp);

    int ret = rknn_init(ctx, model, len, 0, NULL);
    free(model);
    return ret;
}

/* ===================== Run ===================== */
int face_detect_scrfd_run(rknn_context ctx,
                          cv::Mat& img,
                          std::vector<SCRFDResult>& results) {
    results.clear();

    cv::Mat resized;
    float sx = (float)img.cols / g_config.input_width;
    float sy = (float)img.rows / g_config.input_height;
    cv::resize(img, resized,
               {g_config.input_width, g_config.input_height});

    /* ---------- input ---------- */
    rknn_input in;
    memset(&in, 0, sizeof(in));
    in.index = 0;
    in.type  = RKNN_TENSOR_UINT8;
    in.fmt   = RKNN_TENSOR_NHWC;
    in.size  = resized.total() * 3;
    in.buf   = resized.data;
    rknn_inputs_set(ctx, 1, &in);

    rknn_run(ctx, NULL);

    /* ---------- output ---------- */
    rknn_output out[6];
    memset(out, 0, sizeof(out));
    for (int i = 0; i < 6; ++i) {
        out[i].index = i;
        out[i].want_float = 1;
    }
    rknn_outputs_get(ctx, 6, out, NULL);

    std::vector<SCRFDResult> cand;

    for (int s = 0; s < 3; ++s) {
        float* score = (float*)out[s].buf;
        float* box   = (float*)out[s + 3].buf;
        int N = g_anchor_centers[s].size();

        for (int i = 0; i < N; ++i) {
            float conf = sigmoid(score[i]);
            if (conf < g_config.conf_thresh) continue;

            auto& c = g_anchor_centers[s][i];
            float l = box[i * 4 + 0];
            float t = box[i * 4 + 1];
            float r = box[i * 4 + 2];
            float b = box[i * 4 + 3];

            float x1 = c.x - l;
            float y1 = c.y - t;
            float x2 = c.x + r;
            float y2 = c.y + b;

            SCRFDResult res;
            res.box = {x1 * sx, y1 * sy,
                       (x2 - x1) * sx,
                       (y2 - y1) * sy};
            res.score = conf;
            cand.push_back(res);
        }
    }

    rknn_outputs_release(ctx, 6, out);

    /* ---------- NMS ---------- */
    std::sort(cand.begin(), cand.end(),
              [](auto& a, auto& b) { return a.score > b.score; });

    for (auto& c : cand) {
        bool keep = true;
        for (auto& r : results) {
            float iou =
                (c.box & r.box).area() /
                (c.box.area() + r.box.area() -
                 (c.box & r.box).area());
            if (iou > g_config.nms_thresh) {
                keep = false;
                break;
            }
        }
        if (keep) results.push_back(c);
    }
    return results.size();
}
