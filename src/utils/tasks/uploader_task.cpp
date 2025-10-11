#include "uploader_task.h"
extern "C" {
#include "log.h"
}
#include <curl/curl.h>

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

void UploaderTask::enqueue(const cv::Mat& img, int cameraNumber, const std::string& type, const std::string& path) {
    std::lock_guard<std::mutex> lock(mtx);
    queue.push({img.clone(), cameraNumber, type, path, 0});
    cv.notify_one();
}

void UploaderTask::run() {
    while (running) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&]{ return !queue.empty() || !running; });
        if (!running && queue.empty()) break;

        UploadItem item = queue.front(); queue.pop();
        lock.unlock();

        std::string resp = uploadHttp(item.img, item.cameraNumber, item.type, item.path);
        // 假设返回 "code":0 成功，否则重试
        if (resp != "0" && item.retry < 3) {
            item.retry++;
            enqueue(item.img, item.cameraNumber, item.type, item.path);
        }
    }
}

std::string UploaderTask::uploadHttp(const cv::Mat& img, int cameraNumber, const std::string& type, const std::string& path) {
    // 使用 libcurl POST 上传 form-data
    CURL *curl = curl_easy_init();
    if (!curl) return "1";

    std::vector<uchar> buf;
    cv::imencode(".jpg", img, buf);
    curl_mime *form = curl_mime_init(curl);
    curl_mimepart *field;

    // image 文件
    field = curl_mime_addpart(form);
    curl_mime_name(field, "image");
    curl_mime_data(field, reinterpret_cast<char*>(buf.data()), buf.size());
    curl_mime_filename(field, "image.jpg");
    curl_mime_type(field, "image/jpeg");

    // time
    field = curl_mime_addpart(form);
    curl_mime_name(field, "time");
    curl_mime_data(field, std::to_string(time(nullptr)).c_str(), CURL_ZERO_TERMINATED);

    // camerNumber
    field = curl_mime_addpart(form);
    curl_mime_name(field, "camerNumber");
    curl_mime_data(field, std::to_string(cameraNumber).c_str(), CURL_ZERO_TERMINATED);

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

    return (res == CURLE_OK && response_code == 200) ? "0" : "1";
}
