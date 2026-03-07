#include "device_config.h"

#include "json.hpp"
#include "main.h"

#include <fstream>

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
        {"image_path", cfg.uploadImagePath}
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
        {"max_yaw", CAPTURE_MAX_YAW}
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
            loadFromJson(this, j["config"]);
            return true;
        }
        loadFromJson(this, j);
        return true;
    } catch (...) {
        return false;
    }
}
