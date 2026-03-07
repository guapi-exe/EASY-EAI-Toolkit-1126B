#include "main.h"
#include "camera_task.h"
#include "uploader_task.h"
#include "device_config.h"
#include "tcp_client.h"
#include <csignal> 
#include <chrono>
#include <thread>

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

    uploader.setUploadSuccessCallback([&](const UploadItem& item) {
        if (item.type == "face" || item.type == "all") {
            tcpClient.sendCaptureComplete(item.cameraNumber, item.type);
        }
    });

    camera.setPersonEventCallback([&](int personId, const std::string& eventType) {
        if (eventType == "person_appeared") {
            tcpClient.sendPersonAppeared(personId);
        }
    });

    camera.setUploadCallback([&](const cv::Mat& img, int id, const std::string& type) {
        (void)id;
        uploader.enqueue(img, config.cameraNumber, type, config.uploadImagePath);
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
