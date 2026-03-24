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

std::vector<uchar> encode_image_for_upload(const cv::Mat& img) {
    std::vector<uchar> buf;
    cv::imencode(".jpg", img, buf);

    if (buf.size() <= kUploadMaxBytes) {
        return buf;
    }

    // 超过300KB则逐步降低质量重编码
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 85};
    for (int quality = 85; quality >= 35; quality -= 10) {
        params[1] = quality;
        std::vector<uchar> compressed;
        cv::imencode(".jpg", img, compressed, params);
        if (compressed.size() <= kUploadMaxBytes || quality == 35) {
            log_info("UploaderTask: image compressed from %zuKB to %zuKB (quality=%d)",
                     buf.size() / 1024,
                     compressed.size() / 1024,
                     quality);
            return compressed;
        }
    }

    return buf;
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

    cv::Mat work = face;

    if (work.cols < 260) {
        int target_w = std::min(560, std::max(work.cols * 2, 300));
        int target_h = std::max(1, static_cast<int>(work.rows * (target_w / static_cast<float>(work.cols))));
        cv::resize(work, work, cv::Size(target_w, target_h), 0, 0, cv::INTER_CUBIC);
    }

    cv::Mat lab;
    cv::cvtColor(work, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> lab_channels;
    cv::split(lab, lab_channels);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.2, cv::Size(8, 8));
    clahe->apply(lab_channels[0], lab_channels[0]);
    cv::merge(lab_channels, lab);
    cv::cvtColor(lab, work, cv::COLOR_Lab2BGR);

    cv::Mat denoised;
    cv::bilateralFilter(work, denoised, 5, 35, 35);

    cv::Mat gray;
    cv::cvtColor(denoised, gray, cv::COLOR_BGR2GRAY);
    double angle = estimate_blur_angle_deg(gray);

    cv::Mat sharp = denoised.clone();
    cv::Mat kernel = cv::getGaborKernel(cv::Size(9, 9), 2.0, angle * CV_PI / 180.0, 5.0, 0.8, 0.0, CV_32F);
    cv::Mat dir_response;
    cv::filter2D(denoised, dir_response, CV_32F, kernel);
    cv::Mat dir_u8;
    cv::convertScaleAbs(dir_response, dir_u8, 0.08, 0);
    cv::addWeighted(sharp, 1.0, dir_u8, 0.18, 0, sharp);

    cv::Mat blur;
    cv::GaussianBlur(sharp, blur, cv::Size(0, 0), 1.1);
    cv::Mat out;
    cv::addWeighted(sharp, 1.6, blur, -0.6, 0, out);
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
