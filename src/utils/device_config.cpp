#include "device_config.h"

#include "json.hpp"
#include "main.h"

#include <fstream>
#include <algorithm>

using nlohmann::json;

namespace {
static json buildDefaultJson(const DeviceConfig& cfg) {
    json j;
    j["device"] = {
        {"code", cfg.deviceCode},
        {"camera_number", cfg.cameraNumber}
    };
    j["upload"] = {
        {"server", cfg.uploadServer},
        {"image_path", cfg.uploadImagePath},
        {"manual_image_path", cfg.uploadManualImagePath}
    };
    j["tcp"] = {
        {"server_ip", cfg.tcpServerIp},
        {"port", cfg.tcpPort},
        {"heartbeat_interval_sec", cfg.heartbeatIntervalSec},
        {"reconnect_interval_sec", cfg.reconnectIntervalSec}
    };
    j["capture_defaults"] = {
        {"min_clarity", CAPTURE_MIN_CLARITY},
        {"max_motion_ratio", CAPTURE_MAX_MOTION_RATIO},
        {"face_detect_interval", CAPTURE_FACE_DETECT_INTERVAL},
        {"face_input_max_width", CAPTURE_FACE_INPUT_MAX_WIDTH},
        {"min_face_score", CAPTURE_MIN_FACE_SCORE},
        {"headshot_expand_ratio", CAPTURE_HEADSHOT_EXPAND_RATIO},
        {"headshot_down_shift", CAPTURE_HEADSHOT_DOWN_SHIFT},
        {"focus_scale_factor", CAPTURE_FOCUS_SCALE_FACTOR},
        {"min_area_ratio", CAPTURE_MIN_AREA_RATIO},
        {"approach_ratio_pos", CAPTURE_APPROACH_RATIO_POS},
        {"approach_ratio_neg", CAPTURE_APPROACH_RATIO_NEG},
        {"min_track_hits", CAPTURE_MIN_TRACK_HITS},
        {"require_approach", CAPTURE_REQUIRE_APPROACH},
        {"max_yaw", CAPTURE_MAX_YAW},
        {"brightness_black_threshold", cfg.brightnessBlackThreshold}
    };
    j["ircut"] = {
        {"brightness_black_threshold", cfg.brightnessBlackThreshold}
    };
    return j;
}

static void loadFromJson(DeviceConfig* cfg, const json& j) {
    if (j.contains("device") && j["device"].contains("code")) {
        cfg->deviceCode = j["device"]["code"].get<std::string>();
    }
    if (j.contains("device") && j["device"].contains("camera_number")) {
        cfg->cameraNumber = j["device"]["camera_number"].get<int>();
    }

    if (j.contains("upload")) {
        if (j["upload"].contains("server")) {
            cfg->uploadServer = j["upload"]["server"].get<std::string>();
        }
        if (j["upload"].contains("image_path")) {
            cfg->uploadImagePath = j["upload"]["image_path"].get<std::string>();
        }
        if (j["upload"].contains("manual_image_path")) {
            cfg->uploadManualImagePath = j["upload"]["manual_image_path"].get<std::string>();
        }
    }

    if (j.contains("tcp")) {
        if (j["tcp"].contains("server_ip")) {
            cfg->tcpServerIp = j["tcp"]["server_ip"].get<std::string>();
        }
        if (j["tcp"].contains("port")) {
            cfg->tcpPort = j["tcp"]["port"].get<int>();
        }
        if (j["tcp"].contains("heartbeat_interval_sec")) {
            cfg->heartbeatIntervalSec = j["tcp"]["heartbeat_interval_sec"].get<int>();
        }
        if (j["tcp"].contains("reconnect_interval_sec")) {
            cfg->reconnectIntervalSec = j["tcp"]["reconnect_interval_sec"].get<int>();
        }
    }

    if (j.contains("ircut") && j["ircut"].is_object()) {
        if (j["ircut"].contains("brightness_black_threshold")) {
            double v = j["ircut"]["brightness_black_threshold"].get<double>();
            cfg->brightnessBlackThreshold = std::max(0.0, std::min(255.0, v));
        }
    }

    if (j.contains("capture_defaults") && j["capture_defaults"].is_object()) {
        if (j["capture_defaults"].contains("brightness_black_threshold")) {
            double v = j["capture_defaults"]["brightness_black_threshold"].get<double>();
            cfg->brightnessBlackThreshold = std::max(0.0, std::min(255.0, v));
        }
    }
}
}

bool DeviceConfig::save(const std::string& filePath) const {
    std::lock_guard<std::mutex> lock(mtx);
    std::ofstream ofs(filePath, std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) {
        return false;
    }

    ofs << buildDefaultJson(*this).dump(4);
    return true;
}

bool DeviceConfig::loadOrCreate(const std::string& filePath) {
    std::lock_guard<std::mutex> lock(mtx);

    std::ifstream ifs(filePath);
    if (!ifs.is_open()) {
        std::ofstream ofs(filePath, std::ios::out | std::ios::trunc);
        if (!ofs.is_open()) {
            return false;
        }
        ofs << buildDefaultJson(*this).dump(4);
        return true;
    }

    try {
        json j;
        ifs >> j;
        loadFromJson(this, j);
    } catch (...) {
        return false;
    }

    return true;
}

bool DeviceConfig::applyServerConfig(const std::string& jsonPayload) {
    std::lock_guard<std::mutex> lock(mtx);
    try {
        json j = json::parse(jsonPayload);
        if (j.contains("config")) {
            json cfg = j["config"];
            if (cfg.contains("upload") && cfg["upload"].is_object()) {
                // 上传路径由本地运行模式控制(debug/normal)，不接受服务端下发覆盖。
                cfg["upload"].erase("image_path");
                cfg["upload"].erase("manual_image_path");
            }
            loadFromJson(this, cfg);
            return true;
        }
        if (j.contains("upload") && j["upload"].is_object()) {
            j["upload"].erase("image_path");
            j["upload"].erase("manual_image_path");
        }
        loadFromJson(this, j);
        return true;
    } catch (...) {
        return false;
    }
}
