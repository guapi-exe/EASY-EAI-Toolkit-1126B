#pragma once

#include <atomic>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include "device_config.h"

class TcpClient {
public:
    using CommandCallback = std::function<void(const std::string& cmdType, const std::string& payload)>;
    using BrightnessProvider = std::function<double()>;

    TcpClient(DeviceConfig* config, const std::string& configPath);
    ~TcpClient();

    void start();
    void stop();

    void setCommandCallback(CommandCallback cb);
    void setBrightnessProvider(BrightnessProvider provider);

    void sendPersonAppeared(int personId);
    void sendAllPersonLeft();
    void sendCaptureComplete(int personId, const std::string& imageType);

private:
    void workerLoop();
    void heartbeatLoop();
    void tryConnect();
    void closeSocket();
    void flushSendQueue();
    void handleIncomingBuffer(const std::string& incoming);
    void handleMessage(const std::string& line);

    void queueJsonLine(const std::string& jsonLine);
    std::string buildHeartbeatJson();

    static double readCpuUsagePercent();
    static double readMemoryUsagePercent();
    static double readCpuTemperatureC();

private:
    DeviceConfig* config;
    std::string configPath;

    std::atomic<bool> running{false};
    std::atomic<bool> connected{false};
    std::thread worker;
    std::thread heartbeatWorker;

    int sockfd{-1};
    std::string recvBuffer;

    std::mutex ioMtx;
    std::mutex sendQueueMtx;
    std::queue<std::string> sendQueue;

    std::mutex cbMtx;
    CommandCallback commandCallback;
    BrightnessProvider brightnessProvider;
};
