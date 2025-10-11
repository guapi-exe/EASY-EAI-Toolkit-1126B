#pragma once
#include <string>
#include <thread>
#include <atomic>
#include <functional>
#include <chrono>

struct HeartbeatData {
    std::string time;
    std::string power;
    std::string latitude;
    std::string longitude;
    int isNorth;
    int isEast;
};

class HeartbeatTask {
public:
    using Callback = std::function<void(const std::string&)>;

    HeartbeatTask(const std::string& eqCode,
                  const std::string& url,
                  std::chrono::seconds interval = std::chrono::seconds(30));

    ~HeartbeatTask();

    void start();
    void stop();

    void setCallback(Callback cb);
    void updateData(const HeartbeatData& data);

private:
    std::string eqCode;
    std::string url;
    std::chrono::seconds interval;
    std::atomic<bool> running;
    std::thread worker;
    Callback callback;
    HeartbeatData hbData;

    void run();
    void sendHeartbeat();
};
