#include "camera_task_internal.h"

using namespace camera_task_internal;
using namespace cv;
using namespace std;


void CameraTask::candidateEvalLoop(rknn_context faceCtx) {
    while (true) {
        CandidateEvalJob job;
        {
            std::unique_lock<std::mutex> lock(candidateEvalMutex);
            candidateEvalCv.wait(lock, [this]() {
                return !running || !candidateEvalQueue.empty();
            });

            if (!running && candidateEvalQueue.empty()) {
                break;
            }
            if (candidateEvalQueue.empty()) {
                continue;
            }

            job = std::move(candidateEvalQueue.front());
            candidateEvalQueue.pop_front();
        }

        if (!job.personRoi.empty() && job.personRoi.cols > 0 && job.personRoi.rows > 0) {
            DeviceConfig::CaptureDefaults config = getCaptureConfigSnapshot();
            double sceneBrightness = environmentBrightness.load();
            AdaptiveCaptureThresholds adaptiveThresholds =
                buildAdaptiveCaptureThresholds(config, sceneBrightness);
            Mat person_roi_resized;
            int target_width = min(config.faceInputMaxWidth, job.personRoi.cols);
            int target_height = static_cast<int>(job.personRoi.rows * target_width / (float)job.personRoi.cols);

            if (target_width > 0 && target_height > 0) {
                cv::resize(job.personRoi, person_roi_resized, Size(target_width, target_height), 0, 0, cv::INTER_LINEAR);

                std::vector<det> face_result;
                int num_faces = face_detect_run(faceCtx, person_roi_resized, face_result);
                if (num_faces <= 0 || face_result.empty()) {
                    char detail[192];
                    std::snprintf(detail, sizeof(detail), "faces=%d roi=%dx%d area=%.4f motion=%.4f",
                                  num_faces,
                                  job.personRoi.cols,
                                  job.personRoi.rows,
                                  job.areaRatio,
                                  job.motionRatio);
                    logTrackReject("eval", job.trackId, "no_face_detected", detail);
                } else {
                    int best_idx = 0;
                    for (int i = 1; i < num_faces; ++i) {
                        if (face_result[i].score > face_result[best_idx].score) {
                            best_idx = i;
                        }
                    }

                    float face_scale_x = (float)job.personRoi.cols / (float)person_roi_resized.cols;
                    float face_scale_y = (float)job.personRoi.rows / (float)person_roi_resized.rows;

                    det best_face = face_result[best_idx];
                    if (best_face.score >= config.minFaceScore && best_face.landmarks.size() >= 3) {
                        best_face.box.x *= face_scale_x;
                        best_face.box.y *= face_scale_y;
                        best_face.box.width *= face_scale_x;
                        best_face.box.height *= face_scale_y;
                        for (auto& lm : best_face.landmarks) {
                            lm.x *= face_scale_x;
                            lm.y *= face_scale_y;
                        }

                        Rect base_fbox(static_cast<int>(best_face.box.x),
                                      static_cast<int>(best_face.box.y),
                                      static_cast<int>(best_face.box.width),
                                      static_cast<int>(best_face.box.height));
                        base_fbox.x = std::max(0, std::min(base_fbox.x, job.personRoi.cols - 1));
                        base_fbox.y = std::max(0, std::min(base_fbox.y, job.personRoi.rows - 1));
                        base_fbox.width = std::min(base_fbox.width, job.personRoi.cols - base_fbox.x);
                        base_fbox.height = std::min(base_fbox.height, job.personRoi.rows - base_fbox.y);

                        if (base_fbox.width > 0 && base_fbox.height > 0) {
                            int face_short_side = std::min(base_fbox.width, base_fbox.height);
                            int face_area = base_fbox.area();
                            if (face_short_side >= config.minFaceBoxShortSide &&
                                face_area >= config.minFaceBoxArea) {
                                float person_area = static_cast<float>(job.personRoi.cols * job.personRoi.rows);
                                float face_area_ratio = face_area / std::max(1.0f, person_area);
                                float face_width_ratio = static_cast<float>(base_fbox.width) / std::max(1, job.personRoi.cols);
                                float face_center_y_ratio = static_cast<float>(base_fbox.y + base_fbox.height * 0.5f) / std::max(1, job.personRoi.rows);
                                bool face_geometry_ok =
                                    face_area_ratio >= config.faceMinAreaInPerson &&
                                    face_width_ratio >= config.faceMinWidthRatio &&
                                    face_center_y_ratio >= config.faceMinCenterYRatio &&
                                    face_center_y_ratio <= config.faceMaxCenterYRatio;

                                if (face_geometry_ok) {
                                    float margin_left = static_cast<float>(base_fbox.x);
                                    float margin_right = static_cast<float>(job.personRoi.cols - (base_fbox.x + base_fbox.width));
                                    float margin_top = static_cast<float>(base_fbox.y);
                                    float margin_bottom = static_cast<float>(job.personRoi.rows - (base_fbox.y + base_fbox.height));
                                    float margin_x_ratio = std::min(margin_left, margin_right) / std::max(1, base_fbox.width);
                                    float margin_y_ratio = std::min(margin_top, margin_bottom) / std::max(1, base_fbox.height);
                                    float min_margin_ratio = std::min(margin_x_ratio, margin_y_ratio);

                                    float face_edge_occlusion = 0.0f;
                                    if (min_margin_ratio < config.faceEdgeMinMargin) {
                                        face_edge_occlusion =
                                            (config.faceEdgeMinMargin - min_margin_ratio) / config.faceEdgeMinMargin;
                                        face_edge_occlusion = std::max(0.0f, std::min(1.0f, face_edge_occlusion));
                                    }

                                    cv::Point2f left_eye = best_face.landmarks[0];
                                    cv::Point2f right_eye = best_face.landmarks[1];
                                    float dx = right_eye.x - left_eye.x;
                                    if (std::fabs(dx) >= 1e-5f) {
                                        float dy = right_eye.y - left_eye.y;
                                        float roll = std::fabs(std::atan2(dy, dx) * 180.0f / CV_PI);
                                        float eye_center_x = (left_eye.x + right_eye.x) / 2.0f;
                                        float yaw = std::fabs((best_face.landmarks[2].x - eye_center_x) / dx);
                                        bool frontal_ok = isFrontalFace(best_face.landmarks);
                                        bool frontal_relaxed_ok = frontal_ok ||
                                            (roll < 24.0f && yaw < 0.32f && best_face.score >= 0.58f && face_edge_occlusion < 0.40f);
                                        bool weak_frontal_ok = frontal_relaxed_ok ||
                                            (roll < 32.0f && yaw < 0.48f && best_face.score >= config.minFaceScore && face_edge_occlusion < 0.72f);
                                        double reference_clarity = computeFocusMeasure(job.personRoi(base_fbox));
                                        float reference_blur_severity = computeMotionBlurSeverity(job.personRoi(base_fbox));
                                        double current_clarity = reference_clarity;
                                        float current_blur_severity = reference_blur_severity;
                                        cv::Mat fused_person_roi;
                                        const cv::Mat* capture_person_roi = &job.personRoi;
                                        bool multi_frame_fused = false;
                                        int fusion_frame_count = 0;
                                        float fusion_mean_similarity = 0.0f;
                                        bool fusion_enabled =
                                            adaptiveThresholds.lowLightStrength >= kMultiFrameFusionLowLightMinStrength &&
                                            !job.fusionHistory.empty() &&
                                            job.motionRatio <= adaptiveThresholds.maxMotionRejectRatio;
                                        if (fusion_enabled) {
                                            MultiFrameFusionResult fusion_result = fuseTrackHistoryPersonRoi(job.personRoi,
                                                                                                             job.fusionHistory,
                                                                                                             base_fbox,
                                                                                                             adaptiveThresholds.lowLightStrength,
                                                                                                             job.motionRatio);
                                            if (!fusion_result.fused.empty() && fusion_result.acceptedFrames >= 2) {
                                                double fused_clarity = computeFocusMeasure(fusion_result.fused(base_fbox));
                                                float fused_blur_severity = computeMotionBlurSeverity(fusion_result.fused(base_fbox));
                                                bool fusion_better =
                                                    fused_clarity > reference_clarity * 1.06 + 3.0 ||
                                                    (fused_clarity >= reference_clarity * 0.99 &&
                                                     fused_blur_severity + 0.06f < reference_blur_severity) ||
                                                    (fused_blur_severity + 0.10f < reference_blur_severity &&
                                                     fused_clarity >= reference_clarity * 0.95);
                                                if (fusion_better) {
                                                    fused_person_roi = std::move(fusion_result.fused);
                                                    capture_person_roi = &fused_person_roi;
                                                    current_clarity = fused_clarity;
                                                    current_blur_severity = fused_blur_severity;
                                                    multi_frame_fused = true;
                                                    fusion_frame_count = fusion_result.acceptedFrames;
                                                    fusion_mean_similarity = fusion_result.meanSimilarity;
                                                    log_debug("Track %d multi-frame fusion used: frames=%d sim=%.2f clarity=%.1f->%.1f blur=%.2f->%.2f",
                                                              job.trackId,
                                                              fusion_result.acceptedFrames,
                                                              fusion_result.meanSimilarity,
                                                              reference_clarity,
                                                              fused_clarity,
                                                              reference_blur_severity,
                                                              fused_blur_severity);
                                                }
                                            }
                                        }
                                        bool motion_gate_ok = job.motionRatio < adaptiveThresholds.maxMotionRatio;
                                        bool strong_candidate_ok =
                                            (!config.requireFrontalFace || frontal_relaxed_ok) &&
                                            current_clarity > adaptiveThresholds.minClarity &&
                                            current_blur_severity < adaptiveThresholds.maxBlurSeverity &&
                                            yaw < config.maxYaw &&
                                            motion_gate_ok;
                                        bool fallback_candidate_ok =
                                            (!config.requireFrontalFace || weak_frontal_ok) &&
                                            current_clarity > adaptiveThresholds.fallbackMinClarity &&
                                            current_blur_severity < adaptiveThresholds.fallbackMaxBlurSeverity &&
                                            yaw < config.fallbackMaxYaw &&
                                            face_edge_occlusion < config.fallbackMaxFaceEdgeOcclusion &&
                                            motion_gate_ok;

                                        if (strong_candidate_ok || fallback_candidate_ok) {
                                            int crop_w = std::max(1, static_cast<int>(base_fbox.width * config.headshotExpandRatio));
                                            int crop_h = std::max(1, static_cast<int>(base_fbox.height * config.headshotExpandRatio));
                                            int crop_cx = base_fbox.x + base_fbox.width / 2;
                                            int crop_cy = base_fbox.y + base_fbox.height / 2 + static_cast<int>(base_fbox.height * config.headshotDownShift);
                                            int crop_x = crop_cx - crop_w / 2;
                                            int crop_y = crop_cy - crop_h / 2;

                                            crop_x = std::max(0, std::min(job.personRoi.cols - crop_w, crop_x));
                                            crop_y = std::max(0, std::min(job.personRoi.rows - crop_h, crop_y));
                                            Rect fbox(crop_x, crop_y,
                                                      std::min(crop_w, job.personRoi.cols - crop_x),
                                                      std::min(crop_h, job.personRoi.rows - crop_y));

                                            if (fbox.width > 0 && fbox.height > 0) {
                                                float crop_margin_left = static_cast<float>(base_fbox.x - fbox.x) / std::max(1, base_fbox.width);
                                                float crop_margin_right = static_cast<float>((fbox.x + fbox.width) - (base_fbox.x + base_fbox.width)) / std::max(1, base_fbox.width);
                                                float crop_margin_top = static_cast<float>(base_fbox.y - fbox.y) / std::max(1, base_fbox.height);
                                                float crop_margin_bottom = static_cast<float>((fbox.y + fbox.height) - (base_fbox.y + base_fbox.height)) / std::max(1, base_fbox.height);
                                                float crop_min_margin = std::min(std::min(crop_margin_left, crop_margin_right),
                                                                                 std::min(crop_margin_top, crop_margin_bottom));
                                                float required_crop_margin = strong_candidate_ok ?
                                                    config.headshotMinFaceMargin :
                                                    config.fallbackHeadshotMinFaceMargin;
                                                if (crop_min_margin >= required_crop_margin) {
                                                    Mat face_aligned = (*capture_person_roi)(fbox).clone();
                                                    int upper_body_w = std::min(job.personRoi.cols,
                                                        std::max(static_cast<int>(base_fbox.width * config.upperBodyWidthFaceRatio),
                                                                 static_cast<int>(job.personRoi.cols * config.upperBodyMinWidthRatio)));
                                                    int upper_body_h = std::min(job.personRoi.rows,
                                                        std::max(static_cast<int>(base_fbox.height * config.upperBodyHeightFaceRatio),
                                                                 static_cast<int>(job.personRoi.rows * config.upperBodyMinHeightRatio)));
                                                    int upper_body_cx = base_fbox.x + base_fbox.width / 2;
                                                    int upper_body_cy = base_fbox.y + static_cast<int>(base_fbox.height * config.upperBodyCenterYRatio);
                                                    int upper_body_x = std::max(0, std::min(job.personRoi.cols - upper_body_w, upper_body_cx - upper_body_w / 2));
                                                    int upper_body_y = std::max(0, std::min(job.personRoi.rows - upper_body_h,
                                                        upper_body_cy - static_cast<int>(upper_body_h / config.upperBodyTopDivisor)));
                                                    cv::Rect upper_body_box(
                                                        upper_body_x,
                                                        upper_body_y,
                                                        std::min(upper_body_w, job.personRoi.cols - upper_body_x),
                                                        std::min(upper_body_h, job.personRoi.rows - upper_body_y));
                                                    cv::Mat person_aligned = (*capture_person_roi)(upper_body_box).clone();
                                                    float quality_weight, area_weight;
                                                    if (yaw < 0.15f) {
                                                        quality_weight = 0.8f;
                                                        area_weight = 0.35f;
                                                    } else if (yaw < 0.30f) {
                                                        float ratio = (yaw - 0.15f) / 0.15f;
                                                        quality_weight = 0.8f - ratio * 0.3f;
                                                        area_weight = 0.35f - ratio * 0.15f;
                                                    } else if (yaw < 0.50f) {
                                                        float ratio = (yaw - 0.30f) / 0.20f;
                                                        quality_weight = 0.5f - ratio * 0.25f;
                                                        area_weight = 0.2f - ratio * 0.12f;
                                                    } else {
                                                        float ratio = (yaw - 0.50f) / 0.20f;
                                                        quality_weight = 0.25f - ratio * 0.15f;
                                                        area_weight = 0.08f - ratio * 0.05f;
                                                    }

                                                    float area_score = 1.0f / (1.0f + std::fabs(job.areaRatio - config.areaScoreTargetRatio) /
                                                                                        std::max(1e-6f, config.areaScoreTargetRatio));
                                                    float clarity_norm = static_cast<float>(std::min(1.8, current_clarity / std::max(1.0, adaptiveThresholds.minClarity)));
                                                    float person_occ_norm = job.personOcclusion / std::max(1e-6f, config.maxPersonOcclusion);
                                                    person_occ_norm = std::max(0.0f, std::min(1.5f, person_occ_norm));
                                                    float face_edge_occ_norm = face_edge_occlusion / std::max(1e-6f, config.maxFaceEdgeOcclusion);
                                                    face_edge_occ_norm = std::max(0.0f, std::min(1.5f, face_edge_occ_norm));
                                                    float occlusion_penalty = person_occ_norm * 0.7f + face_edge_occ_norm * 0.3f;
                                                    float motion_penalty = std::min(1.25f, job.motionRatio / std::max(1e-6f, adaptiveThresholds.maxMotionRatio));
                                                    float blur_severity_penalty = std::min(1.25f, current_blur_severity / std::max(1e-6f, adaptiveThresholds.maxBlurSeverity));
                                                    float candidate_penalty = strong_candidate_ok ? 0.0f : config.fallbackScorePenalty;
                                                    float clarity_gain = std::max(0.0f, clarity_norm - 1.0f);
                                                    float face_confidence_bonus = std::min(1.0f, std::max(0.0f, best_face.score)) * 90.0f;
                                                    float frontal_bonus = frontal_ok ? 135.0f : (frontal_relaxed_ok ? 75.0f : 20.0f);
                                                    float crop_margin_bonus =
                                                        std::min(1.6f, crop_min_margin / std::max(1e-6f, required_crop_margin)) * 75.0f;
                                                    float face_size_bonus =
                                                        std::min(1.8f, face_area_ratio / std::max(1e-6f, config.faceMinAreaInPerson)) * 45.0f;
                                                    float yaw_penalty =
                                                        std::min(1.25f, yaw / std::max(0.10f, strong_candidate_ok ? config.maxYaw : config.fallbackMaxYaw)) * 60.0f;

                                                    Track::FrameData frame_data;
                                                    frame_data.score = current_clarity * (0.55f + quality_weight * 0.20f) +
                                                                       clarity_norm * 220.0f +
                                                                       clarity_gain * 160.0f +
                                                                       frontal_bonus +
                                                                       face_confidence_bonus +
                                                                       crop_margin_bonus +
                                                                       face_size_bonus +
                                                                       area_score * 1000 * area_weight * 0.22f -
                                                                       occlusion_penalty * config.occlusionScorePenalty * 1.15f -
                                                                       motion_penalty * config.motionScorePenalty * 1.30f -
                                                                       blur_severity_penalty * config.blurSeverityScorePenalty * 1.45f -
                                                                       yaw_penalty -
                                                                       candidate_penalty;
                                                    frame_data.person_roi = person_aligned;
                                                    frame_data.face_roi = face_aligned;
                                                    frame_data.has_face = true;
                                                    frame_data.is_frontal = frontal_ok;
                                                    frame_data.face_pose_level = frontal_ok ? 2 : (frontal_relaxed_ok ? 1 : 0);
                                                    frame_data.strong_candidate = strong_candidate_ok;
                                                    frame_data.yaw_abs = yaw;
                                                    frame_data.clarity = current_clarity;
                                                    frame_data.area_ratio = job.areaRatio;
                                                    frame_data.person_occlusion = job.personOcclusion;
                                                    frame_data.face_edge_occlusion = face_edge_occlusion;
                                                    frame_data.motion_ratio = job.motionRatio;
                                                    frame_data.blur_severity = current_blur_severity;
                                                    add_frame_candidate(job.trackId, frame_data);
                                                    clearTrackReject("eval", job.trackId);

                                                    if (multi_frame_fused) {
                                                        log_debug("Track %d fused candidate accepted: frames=%d sim=%.2f clarity=%.1f blur=%.2f yaw=%.2f edge=%.2f",
                                                                  job.trackId,
                                                                  fusion_frame_count,
                                                                  fusion_mean_similarity,
                                                                  current_clarity,
                                                                  current_blur_severity,
                                                                  yaw,
                                                                  face_edge_occlusion);
                                                    } else if (!strong_candidate_ok) {
                                                        log_debug("Track %d fallback candidate accepted: clarity=%.1f blur=%.2f yaw=%.2f edge=%.2f",
                                                                  job.trackId,
                                                                  current_clarity,
                                                                  current_blur_severity,
                                                                  yaw,
                                                                  face_edge_occlusion);
                                                    }
                                                } else {
                                                    char detail[224];
                                                    std::snprintf(detail, sizeof(detail),
                                                                  "crop_margin=%.2f need=%.2f yaw=%.2f clarity=%.1f",
                                                                  crop_min_margin,
                                                                  required_crop_margin,
                                                                  yaw,
                                                                  current_clarity);
                                                    logTrackReject("eval", job.trackId, "headshot_margin", detail);
                                                }
                                            } else {
                                                char detail[160];
                                                std::snprintf(detail, sizeof(detail), "invalid_headshot_box=%dx%d", fbox.width, fbox.height);
                                                logTrackReject("eval", job.trackId, "headshot_box_invalid", detail);
                                            }
                                        } else {
                                            char detail[320];
                                            std::snprintf(detail, sizeof(detail),
                                                          "frontal=%d relaxed=%d weak=%d yaw=%.2f clarity=%.1f edge=%.2f blur=%.2f motion=%.4f bright=%.1f ll=%.2f",
                                                          frontal_ok ? 1 : 0,
                                                          frontal_relaxed_ok ? 1 : 0,
                                                          weak_frontal_ok ? 1 : 0,
                                                          yaw,
                                                          current_clarity,
                                                          face_edge_occlusion,
                                                          current_blur_severity,
                                                          job.motionRatio,
                                                          sceneBrightness,
                                                          adaptiveThresholds.lowLightStrength);
                                            const char* reason = "quality_gate";
                                            if (config.requireFrontalFace && !weak_frontal_ok) {
                                                reason = "non_frontal";
                                            } else if (!motion_gate_ok) {
                                                reason = "motion_soft_large";
                                            } else if (current_clarity <= adaptiveThresholds.fallbackMinClarity) {
                                                reason = "clarity_low";
                                            } else if (current_blur_severity >= adaptiveThresholds.fallbackMaxBlurSeverity) {
                                                reason = "motion_blur";
                                            } else if (yaw >= config.fallbackMaxYaw) {
                                                reason = "yaw_large";
                                            } else if (face_edge_occlusion >= config.fallbackMaxFaceEdgeOcclusion) {
                                                reason = "edge_occlusion";
                                            }
                                            logTrackReject("eval", job.trackId, reason, detail);
                                        }
                                    } else {
                                        char detail[160];
                                        std::snprintf(detail, sizeof(detail), "eye_dx=%.5f", dx);
                                        logTrackReject("eval", job.trackId, "eye_distance_small", detail);
                                    }
                                } else {
                                    char detail[224];
                                    std::snprintf(detail, sizeof(detail),
                                                  "area_ratio=%.3f width_ratio=%.3f center_y=%.3f",
                                                  face_area_ratio,
                                                  face_width_ratio,
                                                  face_center_y_ratio);
                                    logTrackReject("eval", job.trackId, "face_geometry", detail);
                                }
                            } else {
                                char detail[224];
                                std::snprintf(detail, sizeof(detail), "short=%d min_short=%d area=%d min_area=%d",
                                              face_short_side,
                                              config.minFaceBoxShortSide,
                                              face_area,
                                              config.minFaceBoxArea);
                                logTrackReject("eval", job.trackId, "face_box_small", detail);
                            }
                        } else {
                            char detail[160];
                            std::snprintf(detail, sizeof(detail), "base_fbox=%dx%d", base_fbox.width, base_fbox.height);
                            logTrackReject("eval", job.trackId, "face_box_invalid", detail);
                        }
                    } else {
                        char detail[192];
                        std::snprintf(detail, sizeof(detail), "score=%.2f min=%.2f landmarks=%zu",
                                      best_face.score,
                                      config.minFaceScore,
                                      best_face.landmarks.size());
                        logTrackReject("eval", job.trackId,
                                       best_face.landmarks.size() < 3 ? "landmarks_missing" : "face_score_low",
                                       detail);
                    }
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock(candidateEvalMutex);
            auto it = pendingCandidateEvalByTrack.find(job.trackId);
            if (it != pendingCandidateEvalByTrack.end()) {
                it->second--;
                if (it->second <= 0) {
                    pendingCandidateEvalByTrack.erase(it);
                }
            }
        }
    }
}

