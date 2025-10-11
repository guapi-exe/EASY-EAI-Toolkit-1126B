#pragma once
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include <string>
#include <atomic>

struct UploadItem {
    cv::Mat img;
    int cameraNumber;
    std::string type;
    int retry = 0;
};

class UploaderTask {
public:
    UploaderTask(const std::string& eqCode, const std::string& url);
    ~UploaderTask();

    void start();
    void stop();

    void enqueue(const cv::Mat& img, int cameraNumber, const std::string& type);

private:
    void run();
    std::string uploadHttp(const cv::Mat& img, int cameraNumber, const std::string& type);
    std::string eqCode;
    std::string serverUrl;
    std::queue<UploadItem> queue;
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> running;
    std::thread worker;
};
