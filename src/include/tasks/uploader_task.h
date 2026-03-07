#pragma once
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include <string>
#include <atomic>
#include <functional>

struct UploadItem {
    cv::Mat img;
    int cameraNumber;
    std::string type;
    std::string path;
    int retry = 0;
};

class UploaderTask {
public:
    using UploadSuccessCallback = std::function<void(const UploadItem& item)>;

    UploaderTask(const std::string& eqCode, const std::string& url);
    ~UploaderTask();

    void start();
    void stop();

    void enqueue(const cv::Mat& img, int cameraNumber, const std::string& type, const std::string& path = "");
    void setServerUrl(const std::string& url);
    void setEqCode(const std::string& code);
    void setUploadSuccessCallback(UploadSuccessCallback cb);

private:
    void enqueue(const UploadItem& item);
    void run();
    std::string uploadHttp(const cv::Mat& img, int cameraNumber, const std::string& type, const std::string& path = "");
    std::string eqCode;
    std::string serverUrl;
    std::queue<UploadItem> queue;
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> running;
    std::thread worker;
    UploadSuccessCallback uploadSuccessCallback;
};
