#include "main.h"
#include "camera_task.h"
#include "uploader_task.h"
#include "device_config.h"
#include "tcp_client.h"
#include <atomic>
#include <csignal> 
#include <chrono>
#include <random>
#include <thread>
#include <unordered_map>

extern "C" {
#include "log.h"
}

std::atomic<bool> running(true);

void handleSignal(int) {
    running = false;
}

int main() {
    const std::string configPath = "device_config.json";
    DeviceConfig config;
    if (!config.loadOrCreate(configPath)) {
        log_error("Failed to load or create config file: %s", configPath.c_str());
        return -1;
    }

    UploaderTask uploader(config.deviceCode, config.uploadServer);
    CameraTask camera(PERSON_MODEL_PATH, FACE_MODEL_PATH, CAMERA_INDEX_1);
    TcpClient tcpClient(&config, configPath);

    std::atomic<bool> sleepMode(false);
    std::unordered_map<int, std::string> groupedUniqueCode;

    auto generateUniqueCode12 = []() {
        static thread_local std::mt19937 rng(std::random_device{}());
        static const char kChars[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        std::uniform_int_distribution<int> dist(0, (int)sizeof(kChars) - 2);
        std::string code;
        code.reserve(12);
        for (int i = 0; i < 12; ++i) {
            code.push_back(kChars[dist(rng)]);
        }
        return code;
    };

    uploader.setUploadSuccessCallback([&](const UploadItem& item) {
        if (item.type == "face" || item.type == "all" || item.type == "manual") {
            tcpClient.sendCaptureComplete(item.cameraNumber, item.type);
        }
    });

    camera.setPersonEventCallback([&](int personId, const std::string& eventType) {
        if (eventType == "person_appeared") {
            tcpClient.sendPersonAppeared(personId);
        }
    });

    camera.setUploadCallback([&](const cv::Mat& img, int id, const std::string& type) {
        const std::string& targetPath = (type == "manual")
            ? config.uploadManualImagePath
            : config.uploadImagePath;

        std::string uniqueCode;
        if ((type == "person" || type == "face") && id > 0) {
            auto it = groupedUniqueCode.find(id);
            if (it == groupedUniqueCode.end()) {
                uniqueCode = generateUniqueCode12();
                groupedUniqueCode[id] = uniqueCode;
            } else {
                uniqueCode = it->second;
            }

            // 当前逻辑下同组通常是先person再face，face入队后可清理映射。
            if (type == "face") {
                groupedUniqueCode.erase(id);
            }
        }

        uploader.enqueue(img, config.cameraNumber, type, targetPath, uniqueCode);
    });

    tcpClient.setCommandCallback([&](const std::string& cmdType, const std::string& payload) {
        (void)payload;
        if (cmdType == "sleep") {
            if (!sleepMode) {
                sleepMode = true;
                camera.stop();
                log_info("System enter low-power mode: camera stopped");
            }
            return;
        }

        if (cmdType == "wake") {
            if (sleepMode) {
                sleepMode = false;
                camera.start();
                log_info("System wakeup: camera restarted");
            }
            return;
        }

        if (cmdType == "capture") {
            if (!sleepMode) {
                camera.captureSnapshot();
            }
            return;
        }

        if (cmdType == "config_update") {
            uploader.setServerUrl(config.uploadServer);
            uploader.setEqCode(config.deviceCode);
            log_info("Config updated: upload server=%s, device_code=%s",
                     config.uploadServer.c_str(), config.deviceCode.c_str());
            return;
        }
    });

    uploader.start();
    camera.start();
    tcpClient.start();

    std::signal(SIGINT, handleSignal);

    log_info("System started. Press Ctrl+C to stop.");

    while (running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    log_info("System shutting down...");

    tcpClient.stop();
    camera.stop();
    uploader.stop();
    
    return 0;
}
