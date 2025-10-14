#pragma once
#include <string>
#include <atomic>
#include <thread>

class SerialTask {
public:
    SerialTask(const std::string& port, int baudrate);
    ~SerialTask();

    void start();
    void stop();

private:
    void run();

    std::string port_;
    int baudrate_;
    std::atomic<bool> running_;
    std::thread worker_;
    int fd_;
};