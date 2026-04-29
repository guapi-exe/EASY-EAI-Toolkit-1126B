#include "main.h"
#include "camera_task.h"
#include "uploader_task.h"
#include "device_config.h"
#include "tcp_client.h"
#include "streaming/rtsp_stream_task.h"
#include <algorithm>
#include <atomic>
#include <csignal> 
#include <chrono>
#include <cstdlib>
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

int main(int argc, char** argv) {
    bool debugMode = false;
    bool rtspMode = false;
    RtspStreamOptions rtspOptions;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "debug" || arg == "--debug") {
            debugMode = true;
        } else if (arg == "--rtsp") {
            rtspMode = true;
        } else if (arg == "--rtsp-port" && i + 1 < argc) {
            rtspOptions.port = argv[++i];
        } else if (arg == "--rtsp-path" && i + 1 < argc) {
            rtspOptions.mountPath = argv[++i];
        } else if (arg == "--rtsp-fps" && i + 1 < argc) {
            rtspOptions.fps = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--rtsp-width" && i + 1 < argc) {
            rtspOptions.width = std::max(160, std::atoi(argv[++i]));
        } else if (arg == "--rtsp-height" && i + 1 < argc) {
            rtspOptions.height = std::max(120, std::atoi(argv[++i]));
        } else if (arg == "--rtsp-bitrate" && i + 1 < argc) {
            rtspOptions.bitrateKbps = std::max(256, std::atoi(argv[++i]));
        }
    }
    if (rtspMode && (rtspOptions.mountPath.empty() || rtspOptions.mountPath[0] != '/')) {
        rtspOptions.mountPath.insert(rtspOptions.mountPath.begin(), '/');
    }

    const std::string configPath = "device_config.json";
    DeviceConfig config;
    if (!config.loadOrCreate(configPath)) {
        log_error("Failed to load or create config file: %s", configPath.c_str());
        return -1;
    }
    log_info("Camera config: primary_index=%d secondary_index=%d capture=%dx%d process=%dx%d",
             config.camera.primaryIndex,
             config.camera.secondaryIndex,
             config.camera.captureWidth,
             config.camera.captureHeight,
             config.camera.processWidth,
             config.camera.processHeight);
    if (config.camera.captureWidth != CAMERA_WIDTH ||
        config.camera.captureHeight != CAMERA_HEIGHT ||
        config.camera.processWidth != IMAGE_WIDTH ||
        config.camera.processHeight != IMAGE_HEIGHT) {
        log_warn("Camera resolution config differs from compiled geometry; current pipeline still uses capture=%dx%d process=%dx%d",
                 CAMERA_WIDTH,
                 CAMERA_HEIGHT,
                 IMAGE_WIDTH,
                 IMAGE_HEIGHT);
    }

    if (debugMode) {
        // 调试模式下回退到原自动上传接口，便于联调旧服务。
        if (config.uploadImagePath != "/receive/image/auto") {
            config.uploadImagePath = "/receive/image/auto";
            config.save(configPath);
        }
        log_info("Debug mode enabled: upload path switched to %s", config.uploadImagePath.c_str());
    } else {
        // 非调试模式默认使用 minio 上传接口。
        if (config.uploadImagePath != "/receive/image/auto/minio") {
            config.uploadImagePath = "/receive/image/auto/minio";
            config.save(configPath);
            log_info("Normal mode: upload path switched to %s", config.uploadImagePath.c_str());
        }
    }
    log_info("Upload config: mode=%s server=%s image_path=%s manual_path=%s",
             debugMode ? "debug" : "normal",
             config.uploadServer.c_str(),
             config.uploadImagePath.c_str(),
             config.uploadManualImagePath.c_str());

    UploaderTask uploader(config.deviceCode, config.uploadServer);
    CameraTask camera(PERSON_MODEL_PATH, FACE_MODEL_PATH, config.camera.primaryIndex);
    TcpClient tcpClient(&config, configPath);
    camera.setRuntimeConfig(config);
    if (rtspMode) {
        camera.enableRtspStream(rtspOptions);
        log_info("RTSP overlay stream requested: rtsp://localhost:%s%s",
                 rtspOptions.port.c_str(),
                 rtspOptions.mountPath.c_str());
    }

    std::atomic<bool> sleepMode(false);
    std::unordered_map<int, std::string> groupedUniqueCode;
    std::unordered_map<int, cv::Mat> pendingPersonById;

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
            return;
        }

        if (eventType == "all_person_left") {
            tcpClient.sendAllPersonLeft();
        }
    });

    tcpClient.setBrightnessProvider([&]() {
        return camera.getEnvironmentBrightness();
    });

    camera.setUploadCallback([&](const cv::Mat& img, int id, const std::string& type) {
        const std::string& targetPath = (type == "manual")
            ? config.uploadManualImagePath
            : config.uploadImagePath;

        if (type == "manual") {
            uploader.enqueue(img, config.cameraNumber, type, targetPath, "");
            return;
        }

        std::string uniqueCode;
        if ((type == "person" || type == "person_only" || type == "face") && id > 0) {
            auto it = groupedUniqueCode.find(id);
            if (it == groupedUniqueCode.end()) {
                uniqueCode = generateUniqueCode12();
                groupedUniqueCode[id] = uniqueCode;
            } else {
                uniqueCode = it->second;
            }

            if (type == "person_only") {
                uploader.enqueue(img, config.cameraNumber, "person", targetPath, uniqueCode);
                pendingPersonById.erase(id);
                groupedUniqueCode.erase(id);
                return;
            }

            if (type == "person") {
                // 缓存人形图，等face到来时成对上传。
                pendingPersonById[id] = img.clone();
                return;
            }

            if (type == "face") {
                auto personIt = pendingPersonById.find(id);
                if (personIt != pendingPersonById.end()) {
                    uploader.enqueue(personIt->second, config.cameraNumber, "person", targetPath, uniqueCode);
                    pendingPersonById.erase(personIt);
                } else {
                    log_warn("Face upload id=%d has no paired person image, sending face only", id);
                }

                uploader.enqueue(img, config.cameraNumber, "face", targetPath, uniqueCode);
                groupedUniqueCode.erase(id);
                return;
            }
        }

        // 兜底：未知类型沿原逻辑直接上传。
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
            camera.setRuntimeConfig(config);
            log_info("Config updated: upload server=%s, device_code=%s",
                     config.uploadServer.c_str(), config.deviceCode.c_str());
            log_info("Config updated: ircut brightness_black_threshold=%.1f",
                     config.brightnessBlackThreshold);
            return;
        }
    });

    uploader.start();
    camera.start();
    tcpClient.start();

    std::signal(SIGINT, handleSignal);

    log_info("System started. Press Ctrl+C to stop. mode=%s", debugMode ? "debug" : "normal");

    while (running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    log_info("System shutting down...");

    tcpClient.stop();
    camera.stop();
    uploader.stop();
    
    return 0;
}
