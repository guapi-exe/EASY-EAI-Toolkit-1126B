#pragma once

#include <string>
#include <mutex>
#include "main.h"

struct DeviceConfig {
    std::string deviceCode = "00000001";
    int cameraNumber = DEFAULT_CAMERA_NUMBER;
    std::string uploadServer = "http://101.200.56.225:11100";
    std::string uploadImagePath = "/receive/image/auto/minio";
    std::string uploadManualImagePath = "/receive/image/manual";
    std::string tcpServerIp = "192.168.1.1";
    int tcpPort = 19000;
    int heartbeatIntervalSec = 10;
    int reconnectIntervalSec = 3;
    double brightnessBlackThreshold = CAMERA_BRIGHTNESS_BLACK_THRESHOLD;

    bool loadOrCreate(const std::string& filePath);
    bool save(const std::string& filePath) const;
    bool applyServerConfig(const std::string& jsonPayload);

private:
    mutable std::mutex mtx;
};
