#include "uploader_task.h"
extern "C" {
#include "log.h"
}
#include <curl/curl.h>
#include <random>
#include <cmath>

namespace {
constexpr size_t kUploadMaxBytes = 300 * 1024;
constexpr size_t kUploadQueueMaxSize = 80;
constexpr int kUploadDefaultJpegQuality = 90;
constexpr int kUploadMinLongEdge = 720;
constexpr int kUploadMaxLongEdge = 1280;

std::string generate_unique_code_12() {
    static thread_local std::mt19937 rng(std::random_device{}());
    static const char kChars[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::uniform_int_distribution<int> dist(0, (int)sizeof(kChars) - 2);

    std::string code;
    code.reserve(12);
    for (int i = 0; i < 12; ++i) {
        code.push_back(kChars[dist(rng)]);
    }
    return code;
}

void add_form_field(curl_mime* form, const char* name, const std::string& value) {
    curl_mimepart* field = curl_mime_addpart(form);
    curl_mime_name(field, name);
    curl_mime_data(field, value.c_str(), CURL_ZERO_TERMINATED);
}

cv::Mat resize_to_long_edge(const cv::Mat& img, int target_long_edge, int interpolation) {
    int current_long_edge = std::max(img.cols, img.rows);
    if (img.empty() || current_long_edge <= 0 || current_long_edge <= target_long_edge) {
        return img;
    }

    float scale = target_long_edge / static_cast<float>(current_long_edge);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(), scale, scale, interpolation);
    return resized;
}

std::vector<uchar> encode_jpeg(const cv::Mat& img, int quality) {
    std::vector<uchar> buf;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, quality};
    cv::imencode(".jpg", img, buf, params);
    return buf;
}

std::vector<uchar> encode_image_for_upload(const cv::Mat& img) {
    if (img.empty()) {
        return {};
    }

    std::vector<uchar> original = encode_jpeg(img, kUploadDefaultJpegQuality);
    if (original.size() <= kUploadMaxBytes) {
        return original;
    }

    cv::Mat working = img;
    int original_long_edge = std::max(img.cols, img.rows);
    if (original_long_edge > kUploadMaxLongEdge) {
        working = resize_to_long_edge(working, kUploadMaxLongEdge, cv::INTER_AREA);
    }

    std::vector<uchar> best = encode_jpeg(working, kUploadDefaultJpegQuality);
    static const int kQualities[] = {90, 84, 78, 72, 66, 60, 55};

    while (true) {
        for (int quality : kQualities) {
            std::vector<uchar> compressed = encode_jpeg(working, quality);
            best = compressed;
            if (compressed.size() <= kUploadMaxBytes) {
                log_info("UploaderTask: image encoded from %zuKB to %zuKB (quality=%d, size=%dx%d)",
                         original.size() / 1024,
                         compressed.size() / 1024,
                         quality,
                         working.cols,
                         working.rows);
                return compressed;
            }
        }

        int current_long_edge = std::max(working.cols, working.rows);
        if (current_long_edge <= kUploadMinLongEdge) {
            break;
        }

        int next_long_edge = std::max(kUploadMinLongEdge, static_cast<int>(std::round(current_long_edge * 0.88f)));
        if (next_long_edge >= current_long_edge) {
            break;
        }
        working = resize_to_long_edge(working, next_long_edge, cv::INTER_AREA);
    }

    log_info("UploaderTask: image encoded from %zuKB to %zuKB (quality=%d, size=%dx%d)",
             original.size() / 1024,
             best.size() / 1024,
             55,
             working.cols,
             working.rows);
    return best;
}

double compute_laplacian_variance(const cv::Mat& gray) {
    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_32F);
    cv::Scalar lap_mu, lap_sigma;
    cv::meanStdDev(lap, lap_mu, lap_sigma);
    return lap_sigma[0] * lap_sigma[0];
}

cv::Mat upscale_small_face(const cv::Mat& face) {
    int short_edge = std::min(face.cols, face.rows);
    if (face.empty() || short_edge <= 0 || short_edge >= 220) {
        return face.clone();
    }

    float scale = std::min(1.65f, 220.0f / static_cast<float>(short_edge));
    if (scale <= 1.05f) {
        return face.clone();
    }

    cv::Mat upscaled;
    int interpolation = scale > 1.35f ? cv::INTER_LANCZOS4 : cv::INTER_CUBIC;
    cv::resize(face, upscaled, cv::Size(), scale, scale, interpolation);
    return upscaled;
}

double estimate_blur_angle_deg(const cv::Mat& gray) {
    cv::Mat gx, gy;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);

    double sum_xx = 0.0;
    double sum_yy = 0.0;
    double sum_xy = 0.0;
    for (int y = 0; y < gray.rows; ++y) {
        const float* px = gx.ptr<float>(y);
        const float* py = gy.ptr<float>(y);
        for (int x = 0; x < gray.cols; ++x) {
            double dx = px[x];
            double dy = py[x];
            sum_xx += dx * dx;
            sum_yy += dy * dy;
            sum_xy += dx * dy;
        }
    }

    double theta = 0.5 * std::atan2(2.0 * sum_xy, sum_xx - sum_yy);
    return theta * 180.0 / CV_PI;
}

cv::Mat motion_deblur_enhance_face(const cv::Mat& face) {
    if (face.empty() || face.cols <= 0 || face.rows <= 0) {
        return face;
    }

    cv::Mat gray_in;
    cv::cvtColor(face, gray_in, cv::COLOR_BGR2GRAY);
    double mean_luma_before = cv::mean(gray_in)[0];
    double lap_var = compute_laplacian_variance(gray_in);
    float blur_severity = static_cast<float>(std::max(0.0, std::min(1.0, (120.0 - lap_var) / 120.0)));
    bool needs_upscale = std::min(face.cols, face.rows) < 220;
    bool needs_enhance = needs_upscale || lap_var < 180.0 || mean_luma_before < 105.0;
    if (!needs_enhance) {
        return face.clone();
    }

    cv::Mat work = upscale_small_face(face);

    if (mean_luma_before < 105.0) {
        cv::Mat lab;
        cv::cvtColor(work, lab, cv::COLOR_BGR2Lab);
        std::vector<cv::Mat> lab_channels;
        cv::split(lab, lab_channels);
        double clip_limit = mean_luma_before < 85.0 ? 1.18 : 1.10;
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clip_limit, cv::Size(8, 8));
        clahe->apply(lab_channels[0], lab_channels[0]);
        cv::merge(lab_channels, lab);
        cv::cvtColor(lab, work, cv::COLOR_Lab2BGR);
    }

    cv::Mat work_f;
    work.convertTo(work_f, CV_32FC3);
    cv::Mat base_f;
    cv::GaussianBlur(work_f, base_f, cv::Size(0, 0), 0.75 + 0.20 * blur_severity);
    cv::Mat detail_f = work_f - base_f;
    float detail_gain = 0.14f + 0.10f * blur_severity;
    cv::Mat enhanced_f = work_f + detail_f * detail_gain;

    if (blur_severity > 0.55f) {
        cv::Mat gray_work;
        cv::cvtColor(work, gray_work, cv::COLOR_BGR2GRAY);
        double angle = estimate_blur_angle_deg(gray_work);
        cv::Mat kernel = cv::getGaborKernel(cv::Size(7, 7), 1.8, angle * CV_PI / 180.0, 4.5, 0.9, 0.0, CV_32F);
        cv::Mat directional;
        cv::filter2D(gray_work, directional, CV_32F, kernel);
        cv::normalize(directional, directional, -12.0f, 12.0f, cv::NORM_MINMAX);

        std::vector<cv::Mat> channels;
        cv::split(enhanced_f, channels);
        for (auto& channel : channels) {
            channel += directional * 0.035f;
        }
        cv::merge(channels, enhanced_f);
    }

    cv::Mat enhanced;
    enhanced_f.convertTo(enhanced, CV_8UC3);

    cv::Mat blur;
    cv::GaussianBlur(enhanced, blur, cv::Size(0, 0), 0.70 + 0.15 * blur_severity);
    cv::Mat out;
    float unsharp_amount = 0.14f + 0.08f * blur_severity;
    cv::addWeighted(enhanced, 1.0f + unsharp_amount, blur, -unsharp_amount, 0, out);

    cv::Mat gray_out;
    cv::cvtColor(out, gray_out, cv::COLOR_BGR2GRAY);
    double mean_luma_after = cv::mean(gray_out)[0];
    if (mean_luma_after > 1e-3) {
        double gain = mean_luma_before / mean_luma_after;
        gain = std::max(0.92, std::min(1.03, gain));
        out.convertTo(out, -1, gain, 0);
    }

    return out;
}

cv::Mat motion_deblur_enhance_person(const cv::Mat& person) {
    if (person.empty() || person.cols <= 0 || person.rows <= 0) {
        return person;
    }

    cv::Mat work = person;
    int long_edge = std::max(work.cols, work.rows);
    if (long_edge > 1440) {
        work = resize_to_long_edge(work, 1440, cv::INTER_AREA);
    }

    cv::Mat gray_in;
    cv::cvtColor(work, gray_in, cv::COLOR_BGR2GRAY);
    double mean_luma_before = cv::mean(gray_in)[0];
    double lap_var = compute_laplacian_variance(gray_in);
    float blur_severity = static_cast<float>(std::max(0.0, std::min(1.0, (180.0 - lap_var) / 180.0)));
    if (blur_severity < 0.18f && mean_luma_before >= 95.0) {
        return work.clone();
    }

    cv::Mat work_f;
    work.convertTo(work_f, CV_32FC3);
    cv::Mat base_f;
    cv::GaussianBlur(work_f, base_f, cv::Size(0, 0), 0.85 + 0.30 * blur_severity);
    cv::Mat detail_f = work_f - base_f;
    float detail_gain = 0.10f + 0.08f * blur_severity;
    cv::Mat enhanced_f = work_f + detail_f * detail_gain;

    if (blur_severity > 0.45f) {
        double angle = estimate_blur_angle_deg(gray_in);
        cv::Mat kernel = cv::getGaborKernel(cv::Size(9, 9), 2.0, angle * CV_PI / 180.0, 5.5, 0.9, 0.0, CV_32F);
        cv::Mat directional;
        cv::filter2D(gray_in, directional, CV_32F, kernel);
        cv::normalize(directional, directional, -10.0f, 10.0f, cv::NORM_MINMAX);

        std::vector<cv::Mat> channels;
        cv::split(enhanced_f, channels);
        for (auto& channel : channels) {
            channel += directional * 0.025f;
        }
        cv::merge(channels, enhanced_f);
    }

    cv::Mat enhanced;
    enhanced_f.convertTo(enhanced, CV_8UC3);
    cv::Mat blur;
    cv::GaussianBlur(enhanced, blur, cv::Size(0, 0), 0.9 + 0.2 * blur_severity);
    cv::Mat out;
    float unsharp_amount = 0.10f + 0.06f * blur_severity;
    cv::addWeighted(enhanced, 1.0f + unsharp_amount, blur, -unsharp_amount, 0, out);

    cv::Mat gray_out;
    cv::cvtColor(out, gray_out, cv::COLOR_BGR2GRAY);
    double mean_luma_after = cv::mean(gray_out)[0];
    if (mean_luma_after > 1e-3) {
        double gain = mean_luma_before / mean_luma_after;
        gain = std::max(0.94, std::min(1.04, gain));
        out.convertTo(out, -1, gain, 0);
    }

    return out;
}
}

UploaderTask::UploaderTask(const std::string& eqCode, const std::string& url) : eqCode(eqCode), serverUrl(url), running(false) {}

UploaderTask::~UploaderTask() { stop(); }

void UploaderTask::start() {
    if (running) return;
    running = true;
    worker = std::thread(&UploaderTask::run, this);
}

void UploaderTask::stop() {
    running = false;
    cv.notify_all();
    if (worker.joinable()) worker.join();
}

void UploaderTask::enqueue(const cv::Mat& img,
                          int cameraNumber,
                          const std::string& type,
                          const std::string& path,
                          const std::string& uniqueCode) {
    std::lock_guard<std::mutex> lock(mtx);
    if (queue.size() >= kUploadQueueMaxSize) {
        queue.pop();
        log_warn("UploaderTask: queue full(%zu), drop oldest to keep realtime path smooth", kUploadQueueMaxSize);
    }
    queue.push({img.clone(), cameraNumber, type, path, uniqueCode, 0});
    cv.notify_one();
}

void UploaderTask::enqueue(const UploadItem& item) {
    std::lock_guard<std::mutex> lock(mtx);
    if (queue.size() >= kUploadQueueMaxSize) {
        queue.pop();
        log_warn("UploaderTask: queue full(%zu), drop oldest retry item", kUploadQueueMaxSize);
    }
    queue.push(item);
    cv.notify_one();
}

void UploaderTask::setServerUrl(const std::string& url) {
    std::lock_guard<std::mutex> lock(mtx);
    serverUrl = url;
}

void UploaderTask::setEqCode(const std::string& code) {
    std::lock_guard<std::mutex> lock(mtx);
    eqCode = code;
}

void UploaderTask::setUploadSuccessCallback(UploadSuccessCallback cb) {
    std::lock_guard<std::mutex> lock(mtx);
    uploadSuccessCallback = cb;
}

void UploaderTask::run() {
    while (running) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&]{ return !queue.empty() || !running; });
        if (!running && queue.empty()) break;

        UploadItem item = queue.front(); queue.pop();
        lock.unlock();

        std::string resp = uploadHttp(item.img, item.cameraNumber, item.type, item.path, item.uniqueCode);
        // 假设返回 "code":0 成功，否则重试
        if (resp != "0" && item.retry < 3) {
            item.retry++;
            log_error("upload failed, retrying (%d/3), type=%s, id=%d, path=%s",
                      item.retry, item.type.c_str(), item.cameraNumber, item.path.c_str());
            enqueue(item);
        } else if (resp != "0") {
            log_error("upload failed after max retries, dropped, type=%s, id=%d, path=%s",
                      item.type.c_str(), item.cameraNumber, item.path.c_str());
        } else {
            UploadSuccessCallback cb;
            {
                std::lock_guard<std::mutex> cbLock(mtx);
                cb = uploadSuccessCallback;
            }
            if (cb) {
                cb(item);
            }
        }
    }
}

std::string UploaderTask::uploadHttp(const cv::Mat& img,
                                     int cameraNumber,
                                     const std::string& type,
                                     const std::string& path,
                                     const std::string& uniqueCode) {
    // 使用 libcurl POST 上传 form-data
    CURL *curl = curl_easy_init();
    if (!curl) return "1";

    cv::Mat processed = img;
    if (type == "face") {
        processed = motion_deblur_enhance_face(img);
    } else if (type == "person") {
        processed = motion_deblur_enhance_person(img);
    }

    std::vector<uchar> buf = encode_image_for_upload(processed);
    curl_mime *form = curl_mime_init(curl);
    curl_mimepart *field;

    // image 文件
    field = curl_mime_addpart(form);
    curl_mime_name(field, "image");
    curl_mime_data(field, reinterpret_cast<char*>(buf.data()), buf.size());
    curl_mime_filename(field, "image.jpg");
    curl_mime_type(field, "image/jpeg");

    // time + camerNumber
    add_form_field(form, "time", std::to_string(time(nullptr)));
    add_form_field(form, "camerNumber", std::to_string(cameraNumber));

    // /receive/image/auto/minio 需要附加字段
    if (path.find("/receive/image/auto/minio") != std::string::npos) {
        std::string imageType = "1";
        std::string isHaveFace = "0";

        if (type == "face") {
            imageType = "2";
            isHaveFace = "1";
        }

        add_form_field(form, "imageType", imageType);
        add_form_field(form, "isHaveFace", isHaveFace);
        std::string code = uniqueCode.empty() ? generate_unique_code_12() : uniqueCode;
        add_form_field(form, "uniqueCode", code);
    }

    // eq-code header
    struct curl_slist *headers = nullptr;
    headers = curl_slist_append(headers, ("eq-code: " + eqCode).c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    curl_easy_setopt(curl, CURLOPT_URL, (serverUrl + path).c_str());
    curl_easy_setopt(curl, CURLOPT_MIMEPOST, form);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);

    CURLcode res = curl_easy_perform(curl);
    long response_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);

    curl_slist_free_all(headers);
    curl_mime_free(form);
    curl_easy_cleanup(curl);
    log_debug("response code %ld, curl result %d", response_code, res);
    return (res == CURLE_OK && response_code == 200) ? "0" : "1";
}
