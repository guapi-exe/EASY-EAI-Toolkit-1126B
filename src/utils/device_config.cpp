#include "device_config.h"

#include "json.hpp"
#include "main.h"

#include <fstream>
#include <algorithm>
#include <cmath>

using nlohmann::json;

namespace {
static double roundConfigFloat(double value, int digits = 4) {
    double scale = std::pow(10.0, digits);
    return std::round(value * scale) / scale;
}

static json configFloat(double value, int digits = 4) {
    return json(roundConfigFloat(value, digits));
}

static bool normalizeFloatNumbers(json& node, int digits = 4) {
    bool changed = false;

    if (node.is_object()) {
        for (auto it = node.begin(); it != node.end(); ++it) {
            changed = normalizeFloatNumbers(it.value(), digits) || changed;
        }
        return changed;
    }

    if (node.is_array()) {
        for (auto& item : node) {
            changed = normalizeFloatNumbers(item, digits) || changed;
        }
        return changed;
    }

    if (node.is_number_float()) {
        double current = node.get<double>();
        double rounded = roundConfigFloat(current, digits);
        if (std::fabs(current - rounded) > 1e-9) {
            node = rounded;
            changed = true;
        }
    }

    return changed;
}

static bool patchMissingFields(json& target, const json& defaults) {
    bool changed = false;

    if (!defaults.is_object()) {
        return false;
    }
    if (!target.is_object()) {
        target = json::object();
        changed = true;
    }

    for (auto it = defaults.begin(); it != defaults.end(); ++it) {
        const std::string& key = it.key();
        const json& defaultValue = it.value();

        if (!target.contains(key)) {
            target[key] = defaultValue;
            changed = true;
            continue;
        }

        if (defaultValue.is_object()) {
            changed = patchMissingFields(target[key], defaultValue) || changed;
        }
    }

    return changed;
}

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
        {"min_clarity", configFloat(cfg.captureDefaults.minClarity)},
        {"fallback_min_clarity", configFloat(cfg.captureDefaults.fallbackMinClarity)},
        {"max_motion_ratio", configFloat(cfg.captureDefaults.maxMotionRatio)},
        {"max_motion_reject_ratio", configFloat(cfg.captureDefaults.maxMotionRejectRatio)},
        {"person_detect_interval", cfg.captureDefaults.personDetectInterval},
        {"face_detect_interval", cfg.captureDefaults.faceDetectInterval},
        {"face_input_max_width", cfg.captureDefaults.faceInputMaxWidth},
        {"max_frame_candidates", cfg.captureDefaults.maxFrameCandidates},
        {"candidate_queue_max", cfg.captureDefaults.candidateQueueMax},
        {"candidate_per_track_max_pending", cfg.captureDefaults.candidatePerTrackMaxPending},
        {"person_context_expand_x", configFloat(cfg.captureDefaults.personContextExpandX)},
        {"person_context_expand_top", configFloat(cfg.captureDefaults.personContextExpandTop)},
        {"person_context_expand_bottom", configFloat(cfg.captureDefaults.personContextExpandBottom)},
        {"min_face_score", configFloat(cfg.captureDefaults.minFaceScore)},
        {"headshot_expand_ratio", configFloat(cfg.captureDefaults.headshotExpandRatio)},
        {"headshot_down_shift", configFloat(cfg.captureDefaults.headshotDownShift)},
        {"focus_scale_factor", cfg.captureDefaults.focusScaleFactor},
        {"min_area_ratio", configFloat(cfg.captureDefaults.minAreaRatio)},
        {"near_area_ratio", configFloat(cfg.captureDefaults.nearAreaRatio)},
        {"area_score_target_ratio", configFloat(cfg.captureDefaults.areaScoreTargetRatio)},
        {"approach_ratio_pos", configFloat(cfg.captureDefaults.approachRatioPos)},
        {"approach_ratio_neg", configFloat(cfg.captureDefaults.approachRatioNeg)},
        {"min_track_hits", cfg.captureDefaults.minTrackHits},
        {"require_approach", cfg.captureDefaults.requireApproach},
        {"require_frontal_face", cfg.captureDefaults.requireFrontalFace},
        {"max_yaw", configFloat(cfg.captureDefaults.maxYaw)},
        {"fallback_max_yaw", configFloat(cfg.captureDefaults.fallbackMaxYaw)},
        {"strong_frontal_max_roll", configFloat(cfg.captureDefaults.strongFrontalMaxRoll)},
        {"strong_frontal_max_yaw", configFloat(cfg.captureDefaults.strongFrontalMaxYaw)},
        {"face_min_area_in_person", configFloat(cfg.captureDefaults.faceMinAreaInPerson)},
        {"face_min_width_ratio", configFloat(cfg.captureDefaults.faceMinWidthRatio)},
        {"face_min_center_y_ratio", configFloat(cfg.captureDefaults.faceMinCenterYRatio)},
        {"face_max_center_y_ratio", configFloat(cfg.captureDefaults.faceMaxCenterYRatio)},
        {"min_face_box_short_side", cfg.captureDefaults.minFaceBoxShortSide},
        {"min_face_box_area", cfg.captureDefaults.minFaceBoxArea},
        {"max_person_occlusion", configFloat(cfg.captureDefaults.maxPersonOcclusion)},
        {"max_face_edge_occlusion", configFloat(cfg.captureDefaults.maxFaceEdgeOcclusion)},
        {"face_edge_min_margin", configFloat(cfg.captureDefaults.faceEdgeMinMargin)},
        {"headshot_min_face_margin", configFloat(cfg.captureDefaults.headshotMinFaceMargin)},
        {"fallback_headshot_min_face_margin", configFloat(cfg.captureDefaults.fallbackHeadshotMinFaceMargin)},
        {"fallback_max_face_edge_occlusion", configFloat(cfg.captureDefaults.fallbackMaxFaceEdgeOcclusion)},
        {"upper_body_width_face_ratio", configFloat(cfg.captureDefaults.upperBodyWidthFaceRatio)},
        {"upper_body_height_face_ratio", configFloat(cfg.captureDefaults.upperBodyHeightFaceRatio)},
        {"upper_body_min_width_ratio", configFloat(cfg.captureDefaults.upperBodyMinWidthRatio)},
        {"upper_body_min_height_ratio", configFloat(cfg.captureDefaults.upperBodyMinHeightRatio)},
        {"upper_body_center_y_ratio", configFloat(cfg.captureDefaults.upperBodyCenterYRatio)},
        {"upper_body_top_divisor", configFloat(cfg.captureDefaults.upperBodyTopDivisor)},
        {"motion_score_penalty", configFloat(cfg.captureDefaults.motionScorePenalty)},
        {"occlusion_score_penalty", configFloat(cfg.captureDefaults.occlusionScorePenalty)},
        {"fallback_score_penalty", configFloat(cfg.captureDefaults.fallbackScorePenalty)},
        {"max_blur_severity", configFloat(cfg.captureDefaults.maxBlurSeverity)},
        {"fallback_max_blur_severity", configFloat(cfg.captureDefaults.fallbackMaxBlurSeverity)},
        {"blur_severity_score_penalty", configFloat(cfg.captureDefaults.blurSeverityScorePenalty)},
        {"brightness_sample_interval", cfg.captureDefaults.brightnessSampleInterval},
        {"brightness_white_threshold", configFloat(cfg.captureDefaults.brightnessWhiteThreshold)},
        {"brightness_black_threshold", configFloat(cfg.captureDefaults.brightnessBlackThreshold)}
    };
    j["ircut"] = {
        {"brightness_black_threshold", configFloat(cfg.brightnessBlackThreshold)}
    };
    j["brightness_boost"] = {
        {"target", configFloat(cfg.brightnessBoost.target)},
        {"boost_threshold", configFloat(cfg.brightnessBoost.boostThreshold)},
        {"boost_min_floor", configFloat(cfg.brightnessBoost.boostMinFloor)},
        {"max_alpha", configFloat(cfg.brightnessBoost.maxAlpha)},
        {"max_beta", configFloat(cfg.brightnessBoost.maxBeta)},
        {"gamma", configFloat(cfg.brightnessBoost.gamma)},
        {"dark_blend", configFloat(cfg.brightnessBoost.darkBlend)}
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
            cfg->captureDefaults.brightnessBlackThreshold = cfg->brightnessBlackThreshold;
        }
    }

    if (j.contains("capture_defaults") && j["capture_defaults"].is_object()) {
        const json& capture = j["capture_defaults"];
        auto loadDouble = [&capture](const char* key, double& value) {
            if (capture.contains(key)) value = capture[key].get<double>();
        };
        auto loadFloat = [&capture](const char* key, float& value) {
            if (capture.contains(key)) value = capture[key].get<float>();
        };
        auto loadInt = [&capture](const char* key, int& value) {
            if (capture.contains(key)) value = capture[key].get<int>();
        };
        auto loadBool = [&capture](const char* key, bool& value) {
            if (!capture.contains(key)) {
                return;
            }
            if (capture[key].is_boolean()) {
                value = capture[key].get<bool>();
            } else if (capture[key].is_number_integer()) {
                value = capture[key].get<int>() != 0;
            }
        };

        loadDouble("min_clarity", cfg->captureDefaults.minClarity);
        loadDouble("fallback_min_clarity", cfg->captureDefaults.fallbackMinClarity);
        loadFloat("max_motion_ratio", cfg->captureDefaults.maxMotionRatio);
        loadFloat("max_motion_reject_ratio", cfg->captureDefaults.maxMotionRejectRatio);
        loadInt("person_detect_interval", cfg->captureDefaults.personDetectInterval);
        loadInt("face_detect_interval", cfg->captureDefaults.faceDetectInterval);
        loadInt("face_input_max_width", cfg->captureDefaults.faceInputMaxWidth);
        loadInt("max_frame_candidates", cfg->captureDefaults.maxFrameCandidates);
        loadInt("candidate_queue_max", cfg->captureDefaults.candidateQueueMax);
        loadInt("candidate_per_track_max_pending", cfg->captureDefaults.candidatePerTrackMaxPending);
        loadFloat("person_context_expand_x", cfg->captureDefaults.personContextExpandX);
        loadFloat("person_context_expand_top", cfg->captureDefaults.personContextExpandTop);
        loadFloat("person_context_expand_bottom", cfg->captureDefaults.personContextExpandBottom);
        loadFloat("min_face_score", cfg->captureDefaults.minFaceScore);
        loadFloat("headshot_expand_ratio", cfg->captureDefaults.headshotExpandRatio);
        loadFloat("headshot_down_shift", cfg->captureDefaults.headshotDownShift);
        loadInt("focus_scale_factor", cfg->captureDefaults.focusScaleFactor);
        loadFloat("min_area_ratio", cfg->captureDefaults.minAreaRatio);
        loadFloat("near_area_ratio", cfg->captureDefaults.nearAreaRatio);
        loadFloat("area_score_target_ratio", cfg->captureDefaults.areaScoreTargetRatio);
        loadFloat("approach_ratio_pos", cfg->captureDefaults.approachRatioPos);
        loadFloat("approach_ratio_neg", cfg->captureDefaults.approachRatioNeg);
        loadInt("min_track_hits", cfg->captureDefaults.minTrackHits);
        loadBool("require_approach", cfg->captureDefaults.requireApproach);
        loadBool("require_frontal_face", cfg->captureDefaults.requireFrontalFace);
        loadFloat("max_yaw", cfg->captureDefaults.maxYaw);
        loadFloat("fallback_max_yaw", cfg->captureDefaults.fallbackMaxYaw);
        loadFloat("strong_frontal_max_roll", cfg->captureDefaults.strongFrontalMaxRoll);
        loadFloat("strong_frontal_max_yaw", cfg->captureDefaults.strongFrontalMaxYaw);
        loadFloat("face_min_area_in_person", cfg->captureDefaults.faceMinAreaInPerson);
        loadFloat("face_min_width_ratio", cfg->captureDefaults.faceMinWidthRatio);
        loadFloat("face_min_center_y_ratio", cfg->captureDefaults.faceMinCenterYRatio);
        loadFloat("face_max_center_y_ratio", cfg->captureDefaults.faceMaxCenterYRatio);
        loadInt("min_face_box_short_side", cfg->captureDefaults.minFaceBoxShortSide);
        loadInt("min_face_box_area", cfg->captureDefaults.minFaceBoxArea);
        loadFloat("max_person_occlusion", cfg->captureDefaults.maxPersonOcclusion);
        loadFloat("max_face_edge_occlusion", cfg->captureDefaults.maxFaceEdgeOcclusion);
        loadFloat("face_edge_min_margin", cfg->captureDefaults.faceEdgeMinMargin);
        loadFloat("headshot_min_face_margin", cfg->captureDefaults.headshotMinFaceMargin);
        loadFloat("fallback_headshot_min_face_margin", cfg->captureDefaults.fallbackHeadshotMinFaceMargin);
        loadFloat("fallback_max_face_edge_occlusion", cfg->captureDefaults.fallbackMaxFaceEdgeOcclusion);
        loadFloat("upper_body_width_face_ratio", cfg->captureDefaults.upperBodyWidthFaceRatio);
        loadFloat("upper_body_height_face_ratio", cfg->captureDefaults.upperBodyHeightFaceRatio);
        loadFloat("upper_body_min_width_ratio", cfg->captureDefaults.upperBodyMinWidthRatio);
        loadFloat("upper_body_min_height_ratio", cfg->captureDefaults.upperBodyMinHeightRatio);
        loadFloat("upper_body_center_y_ratio", cfg->captureDefaults.upperBodyCenterYRatio);
        loadFloat("upper_body_top_divisor", cfg->captureDefaults.upperBodyTopDivisor);
        loadFloat("motion_score_penalty", cfg->captureDefaults.motionScorePenalty);
        loadFloat("occlusion_score_penalty", cfg->captureDefaults.occlusionScorePenalty);
        loadFloat("fallback_score_penalty", cfg->captureDefaults.fallbackScorePenalty);
        loadFloat("max_blur_severity", cfg->captureDefaults.maxBlurSeverity);
        loadFloat("fallback_max_blur_severity", cfg->captureDefaults.fallbackMaxBlurSeverity);
        loadFloat("blur_severity_score_penalty", cfg->captureDefaults.blurSeverityScorePenalty);
        loadInt("brightness_sample_interval", cfg->captureDefaults.brightnessSampleInterval);
        loadDouble("brightness_white_threshold", cfg->captureDefaults.brightnessWhiteThreshold);
        loadDouble("brightness_black_threshold", cfg->captureDefaults.brightnessBlackThreshold);
        cfg->brightnessBlackThreshold = std::max(0.0, std::min(255.0, cfg->captureDefaults.brightnessBlackThreshold));
        cfg->captureDefaults.brightnessBlackThreshold = cfg->brightnessBlackThreshold;
    }

    if (j.contains("brightness_boost") && j["brightness_boost"].is_object()) {
        const json& boost = j["brightness_boost"];
        auto loadDouble = [&boost](const char* key, double& value) {
            if (boost.contains(key)) value = boost[key].get<double>();
        };

        loadDouble("target", cfg->brightnessBoost.target);
        loadDouble("boost_threshold", cfg->brightnessBoost.boostThreshold);
        loadDouble("boost_min_floor", cfg->brightnessBoost.boostMinFloor);
        loadDouble("max_alpha", cfg->brightnessBoost.maxAlpha);
        loadDouble("max_beta", cfg->brightnessBoost.maxBeta);
        loadDouble("gamma", cfg->brightnessBoost.gamma);
        loadDouble("dark_blend", cfg->brightnessBoost.darkBlend);
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

        json normalized = buildDefaultJson(*this);
        bool patched = patchMissingFields(j, normalized);
        bool normalizedFloats = normalizeFloatNumbers(j);
        if (patched || normalizedFloats) {
            std::ofstream ofs(filePath, std::ios::out | std::ios::trunc);
            if (!ofs.is_open()) {
                return false;
            }
            ofs << j.dump(4);
        }
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
