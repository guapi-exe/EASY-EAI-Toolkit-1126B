#pragma once

#include <string>
#include <mutex>
#include "main.h"

struct DeviceConfig {
    struct CaptureDefaults {
        double minClarity = CAPTURE_MIN_CLARITY;
        double fallbackMinClarity = CAPTURE_FALLBACK_MIN_CLARITY;
        float maxMotionRatio = CAPTURE_MAX_MOTION_RATIO;
        float maxMotionRejectRatio = CAPTURE_MAX_MOTION_REJECT_RATIO;
        int personDetectInterval = CAPTURE_PERSON_DETECT_INTERVAL;
        int faceDetectInterval = CAPTURE_FACE_DETECT_INTERVAL;
        int faceInputMaxWidth = CAPTURE_FACE_INPUT_MAX_WIDTH;
        int maxFrameCandidates = CAPTURE_MAX_FRAME_CANDIDATES;
        int candidateQueueMax = CAPTURE_CANDIDATE_QUEUE_MAX;
        int candidatePerTrackMaxPending = CAPTURE_CANDIDATE_PER_TRACK_MAX_PENDING;
        float personContextExpandX = CAPTURE_PERSON_CONTEXT_EXPAND_X;
        float personContextExpandTop = CAPTURE_PERSON_CONTEXT_EXPAND_TOP;
        float personContextExpandBottom = CAPTURE_PERSON_CONTEXT_EXPAND_BOTTOM;
        float minFaceScore = CAPTURE_MIN_FACE_SCORE;
        float headshotExpandRatio = CAPTURE_HEADSHOT_EXPAND_RATIO;
        float headshotDownShift = CAPTURE_HEADSHOT_DOWN_SHIFT;
        int focusScaleFactor = CAPTURE_FOCUS_SCALE_FACTOR;
        float minAreaRatio = CAPTURE_MIN_AREA_RATIO;
        float nearAreaRatio = CAPTURE_NEAR_AREA_RATIO;
        float areaScoreTargetRatio = CAPTURE_AREA_SCORE_TARGET_RATIO;
        float approachRatioPos = CAPTURE_APPROACH_RATIO_POS;
        float approachRatioNeg = CAPTURE_APPROACH_RATIO_NEG;
        int minTrackHits = CAPTURE_MIN_TRACK_HITS;
        bool requireApproach = CAPTURE_REQUIRE_APPROACH != 0;
        bool requireFrontalFace = CAPTURE_REQUIRE_FRONTAL_FACE != 0;
        float maxYaw = CAPTURE_MAX_YAW;
        float fallbackMaxYaw = CAPTURE_FALLBACK_MAX_YAW;
        float strongFrontalMaxRoll = CAPTURE_STRONG_FRONTAL_MAX_ROLL;
        float strongFrontalMaxYaw = CAPTURE_STRONG_FRONTAL_MAX_YAW;
        float faceMinAreaInPerson = CAPTURE_FACE_MIN_AREA_IN_PERSON;
        float faceMinWidthRatio = CAPTURE_FACE_MIN_WIDTH_RATIO;
        float faceMinCenterYRatio = CAPTURE_FACE_MIN_CENTER_Y_RATIO;
        float faceMaxCenterYRatio = CAPTURE_FACE_MAX_CENTER_Y_RATIO;
        int minFaceBoxShortSide = CAPTURE_MIN_FACE_BOX_SHORT_SIDE;
        int minFaceBoxArea = CAPTURE_MIN_FACE_BOX_AREA;
        float maxPersonOcclusion = CAPTURE_MAX_PERSON_OCCLUSION;
        float maxFaceEdgeOcclusion = CAPTURE_MAX_FACE_EDGE_OCCLUSION;
        float faceEdgeMinMargin = CAPTURE_FACE_EDGE_MIN_MARGIN;
        float headshotMinFaceMargin = CAPTURE_HEADSHOT_MIN_FACE_MARGIN;
        float fallbackHeadshotMinFaceMargin = CAPTURE_FALLBACK_HEADSHOT_MIN_FACE_MARGIN;
        float fallbackMaxFaceEdgeOcclusion = CAPTURE_FALLBACK_MAX_FACE_EDGE_OCCLUSION;
        float upperBodyWidthFaceRatio = CAPTURE_UPPER_BODY_WIDTH_FACE_RATIO;
        float upperBodyHeightFaceRatio = CAPTURE_UPPER_BODY_HEIGHT_FACE_RATIO;
        float upperBodyMinWidthRatio = CAPTURE_UPPER_BODY_MIN_WIDTH_RATIO;
        float upperBodyMinHeightRatio = CAPTURE_UPPER_BODY_MIN_HEIGHT_RATIO;
        float upperBodyCenterYRatio = CAPTURE_UPPER_BODY_CENTER_Y_RATIO;
        float upperBodyTopDivisor = CAPTURE_UPPER_BODY_TOP_DIVISOR;
        float motionScorePenalty = CAPTURE_MOTION_SCORE_PENALTY;
        float occlusionScorePenalty = CAPTURE_OCCLUSION_SCORE_PENALTY;
        float fallbackScorePenalty = CAPTURE_FALLBACK_SCORE_PENALTY;
        float maxBlurSeverity = CAPTURE_MAX_BLUR_SEVERITY;
        float fallbackMaxBlurSeverity = CAPTURE_FALLBACK_MAX_BLUR_SEVERITY;
        float blurSeverityScorePenalty = CAPTURE_BLUR_SEVERITY_SCORE_PENALTY;
        double lowLightBrightnessThreshold = CAPTURE_LOW_LIGHT_BRIGHTNESS_THRESHOLD;
        double lowLightBrightnessFloor = CAPTURE_LOW_LIGHT_BRIGHTNESS_FLOOR;
        float lowLightMotionRatioScale = CAPTURE_LOW_LIGHT_MOTION_RATIO_SCALE;
        float lowLightMotionRejectRatioScale = CAPTURE_LOW_LIGHT_MOTION_REJECT_RATIO_SCALE;
        float lowLightMaxBlurSeverityScale = CAPTURE_LOW_LIGHT_MAX_BLUR_SEVERITY_SCALE;
        float lowLightFallbackMaxBlurSeverityScale = CAPTURE_LOW_LIGHT_FALLBACK_MAX_BLUR_SEVERITY_SCALE;
        float lowLightMinClarityScale = CAPTURE_LOW_LIGHT_MIN_CLARITY_SCALE;
        float lowLightFallbackMinClarityScale = CAPTURE_LOW_LIGHT_FALLBACK_MIN_CLARITY_SCALE;
        int brightnessSampleInterval = CAMERA_BRIGHTNESS_SAMPLE_INTERVAL;
        double brightnessWhiteThreshold = CAMERA_BRIGHTNESS_WHITE_THRESHOLD;
        double brightnessBlackThreshold = CAMERA_BRIGHTNESS_BLACK_THRESHOLD;
    };

    struct BrightnessBoostConfig {
        double target = CAMERA_BRIGHTNESS_TARGET;
        double boostThreshold = CAMERA_BRIGHTNESS_BOOST_THRESHOLD;
        double boostMinFloor = CAMERA_BRIGHTNESS_BOOST_MIN_FLOOR;
        double maxAlpha = CAMERA_BRIGHTNESS_MAX_ALPHA;
        double maxBeta = CAMERA_BRIGHTNESS_MAX_BETA;
        double gamma = CAMERA_BRIGHTNESS_GAMMA;
        double darkBlend = CAMERA_BRIGHTNESS_DARK_BLEND;
    };

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
    CaptureDefaults captureDefaults;
    BrightnessBoostConfig brightnessBoost;

    bool loadOrCreate(const std::string& filePath);
    bool save(const std::string& filePath) const;
    bool applyServerConfig(const std::string& jsonPayload);

private:
    mutable std::mutex mtx;
};
