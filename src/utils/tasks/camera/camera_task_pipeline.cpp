#include "camera_task_internal.h"

using namespace camera_task_internal;
using namespace cv;
using namespace std;


void CameraTask::processFrame(const Mat& frame, rknn_context personCtx) {
    static int personDetectCounter = 0;
    static std::vector<Track> cachedTracks;
    DeviceConfig::CaptureDefaults config = getCaptureConfigSnapshot();
    double sceneBrightness = environmentBrightness.load();
    AdaptiveCaptureThresholds adaptiveThresholds =
        buildAdaptiveCaptureThresholds(config, sceneBrightness);

    const float inv_diag_720p = 1.0f / std::sqrt((float)IMAGE_WIDTH * IMAGE_WIDTH + (float)IMAGE_HEIGHT * IMAGE_HEIGHT);

    float scale_x = (float)IMAGE_WIDTH / (float)CAMERA_WIDTH;   // 1280/3840 = 0.333
    float scale_y = (float)IMAGE_HEIGHT / (float)CAMERA_HEIGHT; // 720/2160 = 0.333
    
    // Keep software resize here. The RGA path was slower in practice.
    
    Mat resized_frame;
    cv::resize(frame, resized_frame, Size(IMAGE_WIDTH, IMAGE_HEIGHT), 0, 0, cv::INTER_LINEAR);

    int personDetectInterval = std::max(1, config.personDetectInterval);
    bool runPersonDetect = (personDetectCounter % personDetectInterval == 0) || cachedTracks.empty();
    personDetectCounter++;

    if (runPersonDetect) {
        detect_result_group_t detect_result_group;
        person_detect_run(personCtx, resized_frame, &detect_result_group);

        vector<Detection> dets;
        dets.reserve(detect_result_group.count);
        for (int i = 0; i < detect_result_group.count; i++) {
            detect_result_t& d = detect_result_group.results[i];
            if (d.prop < 0.7f) continue;

            Rect roi_720p(max(0, d.box.left), max(0, d.box.top),
                          min(IMAGE_WIDTH - 1, d.box.right) - max(0, d.box.left),
                          min(IMAGE_HEIGHT - 1, d.box.bottom) - max(0, d.box.top));
            if (roi_720p.width <= 0 || roi_720p.height <= 0) continue;

            Detection det;
            det.roi = resized_frame(roi_720p);
            det.x1 = roi_720p.x;
            det.y1 = roi_720p.y;
            det.x2 = roi_720p.x + roi_720p.width;
            det.y2 = roi_720p.y + roi_720p.height;
            det.prop = d.prop;
            dets.push_back(det);
        }

        nmsDetections(dets, 0.55f);
        cachedTracks = sort_update(dets);
    }

    vector<Track> tracks = cachedTracks;
    std::unordered_set<int> activeTrackIds;

    std::unordered_map<int, cv::Rect> trackBoxes720p;
    trackBoxes720p.reserve(tracks.size());
    for (const auto& tr : tracks) {
        cv::Rect2f stable_box = selectTrackRect720p(tr);
        cv::Rect box((int)stable_box.x, (int)stable_box.y, (int)stable_box.width, (int)stable_box.height);
        if (box.width > 0 && box.height > 0) {
            trackBoxes720p.emplace(tr.id, box);
        }
    }

    std::unordered_map<int, float> trackOcclusionRatio;
    trackOcclusionRatio.reserve(trackBoxes720p.size());
    for (const auto& kv_a : trackBoxes720p) {
        float max_overlap = 0.0f;
        for (const auto& kv_b : trackBoxes720p) {
            if (kv_a.first == kv_b.first) {
                continue;
            }
            float overlap = rect_overlap_ratio_on_a(kv_a.second, kv_b.second);
            if (overlap > max_overlap) {
                max_overlap = overlap;
            }
        }
        trackOcclusionRatio[kv_a.first] = max_overlap;
    }

    if (!tracks.empty()) {
        candidateRoundRobinOffset %= tracks.size();
    } else {
        candidateRoundRobinOffset = 0;
    }

    size_t track_count = tracks.size();
    for (size_t index = 0; index < track_count; ++index) {
        auto& t = tracks[(candidateRoundRobinOffset + index) % track_count];
        activeTrackIds.insert(t.id);

        cv::Rect2f stable_bbox_720p = selectTrackRect720p(t);
        Rect bbox_720p((int)stable_bbox_720p.x, (int)stable_bbox_720p.y,
                       (int)stable_bbox_720p.width, (int)stable_bbox_720p.height);
        if (bbox_720p.width <=0 || bbox_720p.height <=0) continue;

        cv::Point2f curr_center_720p(bbox_720p.x + bbox_720p.width * 0.5f,
                                     bbox_720p.y + bbox_720p.height * 0.5f);
        float motion_ratio = 0.0f;
        auto prev_it = lastTrackCenters.find(t.id);
        if (prev_it != lastTrackCenters.end()) {
            float pixel_motion = cv::norm(curr_center_720p - prev_it->second);
            motion_ratio = pixel_motion * inv_diag_720p;
        }
        lastTrackCenters[t.id] = curr_center_720p;

        if (t.hits < config.minTrackHits) {
            char detail[160];
            std::snprintf(detail, sizeof(detail), "hits=%d min_hits=%d", t.hits, config.minTrackHits);
            logTrackReject("gate", t.id, "track_unstable", detail);
            continue;
        }

        if (reportedPersonIds.find(t.id) == reportedPersonIds.end()) {
            reportedPersonIds.insert(t.id);
            if (personEventCallback) {
                personEventCallback(t.id, "person_appeared");
            }
        }
        
        // Map the stable 720p track box back to the full-resolution frame.
        int orig_x = static_cast<int>(bbox_720p.x / scale_x);
        int orig_y = static_cast<int>(bbox_720p.y / scale_y);
        int orig_width = static_cast<int>(bbox_720p.width / scale_x);
        int orig_height = static_cast<int>(bbox_720p.height / scale_y);
        
        orig_x = max(0, min(CAMERA_WIDTH - orig_width, orig_x));
        orig_y = max(0, min(CAMERA_HEIGHT - orig_height, orig_y));
        orig_width = min(CAMERA_WIDTH - orig_x, orig_width);
        orig_height = min(CAMERA_HEIGHT - orig_y, orig_height);
        
        if (orig_width <= 0 || orig_height <= 0) continue;
        
        int expand_x = static_cast<int>(orig_width * config.personContextExpandX);
        int expand_top = static_cast<int>(orig_height * config.personContextExpandTop);
        int expand_bottom = static_cast<int>(orig_height * config.personContextExpandBottom);
        int expanded_x = std::max(0, orig_x - expand_x);
        int expanded_y = std::max(0, orig_y - expand_top);
        int expanded_right = std::min(CAMERA_WIDTH, orig_x + orig_width + expand_x);
        int expanded_bottom = std::min(CAMERA_HEIGHT, orig_y + orig_height + expand_bottom);
        Rect bbox_4k(expanded_x,
                 expanded_y,
                 std::max(1, expanded_right - expanded_x),
                 std::max(1, expanded_bottom - expanded_y));

        Mat person_roi = frame(bbox_4k);
        if (person_roi.empty() || person_roi.cols <= 0 || person_roi.rows <= 0) {
            char detail[192];
            std::snprintf(detail, sizeof(detail), "roi=%dx%d bbox4k=%dx%d",
                          person_roi.cols,
                          person_roi.rows,
                          bbox_4k.width,
                          bbox_4k.height);
            logTrackReject("gate", t.id, "person_roi_invalid", detail);
            continue;
        }

        std::vector<cv::Mat> fusion_history;
        auto history_it = trackPersonRoiHistory.find(t.id);
        if (history_it != trackPersonRoiHistory.end()) {
            fusion_history.reserve(history_it->second.size());
            for (const auto& hist_roi : history_it->second) {
                if (!hist_roi.empty()) {
                    fusion_history.push_back(hist_roi.clone());
                }
            }
        }
        auto& roi_history = trackPersonRoiHistory[t.id];
        roi_history.push_back(person_roi.clone());
        while (roi_history.size() > kMultiFrameFusionHistorySize) {
            roi_history.pop_front();
        }

        float current_area_4k = bbox_4k.width * bbox_4k.height;
        float area_ratio = current_area_4k / (CAMERA_WIDTH * CAMERA_HEIGHT);

        float area_trend_ratio = 0.0f;
        if (t.bbox_history.size() >= 5) {
            float area_now = t.bbox_history.back();
            float area_prev = t.bbox_history[t.bbox_history.size() - 5];
            area_trend_ratio = (area_now - area_prev) / (area_prev + 1e-6f);

            // Update the coarse approach flag before the stricter hysteresis logic.
            if (area_trend_ratio > config.approachRatioPos) {
                t.is_approaching = true;
            } else if (area_trend_ratio < config.approachRatioNeg) {
                t.is_approaching = false;
            }
        }

        bool near_ok = area_ratio >= config.nearAreaRatio;
        bool approach_ok = t.is_approaching || near_ok || !config.requireApproach;
        auto& approachState = trackApproachStates[t.id];
        float bbox_jitter = t.bbox_jitter;
        bool trend_ready = t.bbox_history.size() >= 4;
        bool history_advanced = t.bbox_history.size() != approachState.lastHistorySize;
        bool jitter_freeze = bbox_jitter >= kApproachJitterFreezeThreshold && !near_ok;

        area_trend_ratio = computeTrackAreaTrendRatio(t);
        if (trend_ready && history_advanced && !jitter_freeze) {
            if (area_trend_ratio > config.approachRatioPos) {
                approachState.positiveHits = std::min(approachState.positiveHits + 1,
                                                      kApproachPositiveFramesRequired + 1);
                if (approachState.negativeHits > 0) {
                    approachState.negativeHits--;
                }
            } else if (area_trend_ratio < config.approachRatioNeg) {
                approachState.negativeHits = std::min(approachState.negativeHits + 1,
                                                      kApproachNegativeFramesRequired + 1);
                if (approachState.positiveHits > 0) {
                    approachState.positiveHits--;
                }
            } else {
                approachState.positiveHits = std::max(0, approachState.positiveHits - 1);
                approachState.negativeHits = std::max(0, approachState.negativeHits - 1);
            }
        } else if (history_advanced && jitter_freeze) {
            approachState.positiveHits = std::max(0, approachState.positiveHits - 1);
            approachState.negativeHits = std::max(0, approachState.negativeHits - 1);
        }

        if (approachState.positiveHits >= kApproachPositiveFramesRequired) {
            approachState.isApproaching = true;
        }
        int negative_required = near_ok ? (kApproachNegativeFramesRequired + 1)
                                        : kApproachNegativeFramesRequired;
        if (approachState.negativeHits >= negative_required) {
            approachState.isApproaching = false;
        }

        approachState.lastTrend = area_trend_ratio;
        approachState.lastJitter = bbox_jitter;
        approachState.lastAreaRatio = area_ratio;
        approachState.lastHistorySize = t.bbox_history.size();
        t.is_approaching = approachState.isApproaching;
        approach_ok = t.is_approaching || near_ok || !config.requireApproach;
        bool moving_away = trend_ready &&
                           !jitter_freeze &&
                           area_trend_ratio < config.approachRatioNeg &&
                           approachState.negativeHits >= 2 &&
                           !near_ok;
        bool person_captured = capturedPersonIds.find(t.id) != capturedPersonIds.end();
        bool face_captured = capturedFaceIds.find(t.id) != capturedFaceIds.end();
        if (person_captured && face_captured) {
            logTrackReject("gate", t.id, "already_captured", "person and face already captured");
            continue;
        }
        if (bbox_jitter > kApproachJitterRejectThreshold && !near_ok) {
            char detail[224];
            std::snprintf(detail, sizeof(detail), "jitter=%.3f trend=%.3f area=%.4f near=%d",
                          bbox_jitter,
                          area_trend_ratio,
                          area_ratio,
                          near_ok ? 1 : 0);
            logTrackReject("gate", t.id, "bbox_jitter", detail);
            continue;
        }
        if (!approach_ok) {
            char detail[224];
            std::snprintf(detail, sizeof(detail),
                          "approaching=%d near=%d trend=%.3f jitter=%.3f pos=%d neg=%d area=%.4f",
                          t.is_approaching ? 1 : 0,
                          near_ok ? 1 : 0,
                          area_trend_ratio,
                          bbox_jitter,
                          approachState.positiveHits,
                          approachState.negativeHits,
                          area_ratio);
            logTrackReject("gate", t.id, "not_approaching", detail);
            continue;
        }
        if (moving_away) {
            char detail[192];
            std::snprintf(detail, sizeof(detail), "trend=%.3f jitter=%.3f neg=%d area=%.4f",
                          area_trend_ratio,
                          bbox_jitter,
                          approachState.negativeHits,
                          area_ratio);
            logTrackReject("gate", t.id, "moving_away", detail);
            continue;
        }
        if (area_ratio <= config.minAreaRatio) {
            char detail[192];
            std::snprintf(detail, sizeof(detail), "area=%.4f min=%.4f near=%.4f",
                          area_ratio,
                          config.minAreaRatio,
                          config.nearAreaRatio);
            logTrackReject("gate", t.id, "too_far", detail);
            continue;
        }
        if (motion_ratio > adaptiveThresholds.maxMotionRejectRatio) {
            char detail[224];
            std::snprintf(detail, sizeof(detail), "motion=%.4f max=%.4f area=%.4f bright=%.1f ll=%.2f",
                          motion_ratio,
                          adaptiveThresholds.maxMotionRejectRatio,
                          area_ratio,
                          sceneBrightness,
                          adaptiveThresholds.lowLightStrength);
            logTrackReject("gate", t.id, "motion_large", detail);
            continue;
        }

        float person_occlusion = 0.0f;
        auto occ_it = trackOcclusionRatio.find(t.id);
        if (occ_it != trackOcclusionRatio.end()) {
            person_occlusion = occ_it->second;
        }

        CandidateEvalJob job;
        job.trackId = t.id;
        job.personRoi = person_roi.clone();
        job.fusionHistory = std::move(fusion_history);
        job.areaRatio = area_ratio;
        job.personOcclusion = person_occlusion;
        job.motionRatio = motion_ratio;
        if (enqueueCandidateEvaluation(std::move(job))) {
            clearTrackReject("gate", t.id);
        }
    }

    if (track_count > 0) {
        candidateRoundRobinOffset = (candidateRoundRobinOffset + 1) % track_count;
    }

    for (auto it = lastTrackCenters.begin(); it != lastTrackCenters.end(); ) {
        if (activeTrackIds.find(it->first) == activeTrackIds.end()) {
            it = lastTrackCenters.erase(it);
        } else {
            ++it;
        }
    }

    for (auto it = trackApproachStates.begin(); it != trackApproachStates.end(); ) {
        if (activeTrackIds.find(it->first) == activeTrackIds.end()) {
            it = trackApproachStates.erase(it);
        } else {
            ++it;
        }
    }

    for (auto it = trackPersonRoiHistory.begin(); it != trackPersonRoiHistory.end(); ) {
        if (activeTrackIds.find(it->first) == activeTrackIds.end()) {
            it = trackPersonRoiHistory.erase(it);
        } else {
            ++it;
        }
    }

    {
        std::lock_guard<std::mutex> lock(candidateEvalMutex);
        for (auto it = pendingCandidateEvalByTrack.begin(); it != pendingCandidateEvalByTrack.end(); ) {
            if (activeTrackIds.find(it->first) == activeTrackIds.end() && it->second <= 0) {
                it = pendingCandidateEvalByTrack.erase(it);
            } else {
                ++it;
            }
        }
    }

    for (auto it = reportedPersonIds.begin(); it != reportedPersonIds.end(); ) {
        if (activeTrackIds.find(*it) == activeTrackIds.end()) {
            it = reportedPersonIds.erase(it);
        } else {
            ++it;
        }
    }

    bool hasPersons = !activeTrackIds.empty();
    if (hadPersonsInScene && !hasPersons) {
        if (personEventCallback) {
            personEventCallback(-1, "all_person_left");
        }
    }
    hadPersonsInScene = hasPersons;
}
