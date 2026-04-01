#pragma once

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#define CAMERA_WIDTH    2688
#define CAMERA_HEIGHT   1520
#define IMAGE_WIDTH     1280
#define IMAGE_HEIGHT    720
#define CAMERA_INDEX_1  11
#define CAMERA_INDEX_2  51
#define DEFAULT_CAMERA_NUMBER 1
#define CAMERA_FORMAT   RK_FORMAT_BGR_888  
#define IMGRATIO        3
#define IMAGE_SIZE      (CAMERA_WIDTH*CAMERA_HEIGHT*IMGRATIO)
#define MAX_MISSED      45

#define PERSON_MODEL_PATH   "person_detect.model"
#define FACE_MODEL_PATH     "face_detect.model"
#define NAFNET_TINY_MODEL_PATH "nafnet_tiny.rknn"

#define RETIAN_MODEL_TYPE   0 
#define RETIAN_INPUT_H      480
#define RETIAN_INPUT_W      640
#define RETIAN_CONF_THRESH  0.7f
#define RETIAN_NMS_THRESH   0.4f

// 抓拍与性能调优参数
// 清晰度阈值：正常抓拍使用严格阈值，fallback 路径允许较弱清晰度但会附带惩罚。
#define CAPTURE_MIN_CLARITY              110.0
#define CAPTURE_FALLBACK_MIN_CLARITY     55.0
// 运动阈值：motion_ratio 越大说明目标移动越快；超过 reject 阈值直接淘汰。
#define CAPTURE_MAX_MOTION_RATIO         0.014f
#define CAPTURE_MAX_MOTION_REJECT_RATIO  0.038f
// 推理节流：每隔 N 帧做人/脸检测，平衡实时性与 CPU/NPU 开销。
#define CAPTURE_PERSON_DETECT_INTERVAL   2
#define CAPTURE_FACE_DETECT_INTERVAL    2
// 人脸检测前将 ROI 限制到该宽度以内，降低 face detect 开销。
#define CAPTURE_FACE_INPUT_MAX_WIDTH    640
// 缓存与队列限制：防止轨迹候选过多导致内存占用和延迟累积。
#define CAPTURE_MAX_FRAME_CANDIDATES    48
#define CAPTURE_CANDIDATE_QUEUE_MAX     128
#define CAPTURE_CANDIDATE_PER_TRACK_MAX_PENDING 6
// 人体上下文扩展：在人框基础上向四周扩一点，避免脸框过紧或裁掉头顶/下巴。
#define CAPTURE_PERSON_CONTEXT_EXPAND_X 0.12f
#define CAPTURE_PERSON_CONTEXT_EXPAND_TOP 0.18f
#define CAPTURE_PERSON_CONTEXT_EXPAND_BOTTOM 0.10f
// 人脸置信度最低阈值。
#define CAPTURE_MIN_FACE_SCORE          0.55f
// 头肩图裁切比例：以脸框为基准向外扩并略微下移，得到更自然的人像构图。
#define CAPTURE_HEADSHOT_EXPAND_RATIO   1.80f
#define CAPTURE_HEADSHOT_DOWN_SHIFT     0.20f
// 清晰度评估时缩放倍数，放大后测 Laplacian 可提升小脸区域稳定性。
#define CAPTURE_FOCUS_SCALE_FACTOR      3
// 面积阈值：过滤过远过小的人体；near ratio 可用于近距离优先策略。
#define CAPTURE_MIN_AREA_RATIO          0.025f
#define CAPTURE_NEAR_AREA_RATIO         0.065f
// 面积评分中心值：抓拍时更偏好该人体面积占比附近的候选，过大或过小都会降分。
#define CAPTURE_AREA_SCORE_TARGET_RATIO 0.6f
// 接近趋势判断：目标面积变化为正且超过阈值视为“走近镜头”。
#define CAPTURE_APPROACH_RATIO_POS      0.16f
#define CAPTURE_APPROACH_RATIO_NEG     -0.16f
// 轨迹最少命中次数，避免刚出现的瞬时误检直接触发抓拍。
#define CAPTURE_MIN_TRACK_HITS          3
// 是否要求目标处于接近状态、是否要求正脸。
#define CAPTURE_REQUIRE_APPROACH        1
#define CAPTURE_REQUIRE_FRONTAL_FACE    1
// 姿态阈值：正常路径更严格，fallback 路径放宽 yaw。
#define CAPTURE_MAX_YAW                 0.8f
#define CAPTURE_FALLBACK_MAX_YAW        0.95f
// 强正脸阈值：用于优先选择更“正”的脸，抑制侧脸抓拍。
#define CAPTURE_STRONG_FRONTAL_MAX_ROLL 18.0f
#define CAPTURE_STRONG_FRONTAL_MAX_YAW  0.22f
// 脸框几何约束：限制脸在人框中的比例、宽度和垂直位置，过滤误检。
#define CAPTURE_FACE_MIN_AREA_IN_PERSON 0.018f
#define CAPTURE_FACE_MIN_WIDTH_RATIO    0.085f
#define CAPTURE_FACE_MIN_CENTER_Y_RATIO 0.10f
#define CAPTURE_FACE_MAX_CENTER_Y_RATIO 0.58f
// 脸框尺寸阈值：过滤太小的人脸，避免上传不可用的小图。
#define CAPTURE_MIN_FACE_BOX_SHORT_SIDE 76
#define CAPTURE_MIN_FACE_BOX_AREA       9000
// 遮挡阈值：人体或脸边缘被遮挡太严重时放弃该候选。
#define CAPTURE_MAX_PERSON_OCCLUSION    0.45f
#define CAPTURE_MAX_FACE_EDGE_OCCLUSION 0.55f
// 脸到边缘的最小安全边距，避免脸贴边导致后续裁切不完整。
#define CAPTURE_FACE_EDGE_MIN_MARGIN    0.08f
#define CAPTURE_HEADSHOT_MIN_FACE_MARGIN 0.08f
#define CAPTURE_FALLBACK_HEADSHOT_MIN_FACE_MARGIN 0.01f
#define CAPTURE_FALLBACK_MAX_FACE_EDGE_OCCLUSION 0.88f
// 上半身构图约束：控制头肩图宽高比例和脸在图中的垂直位置。
#define CAPTURE_UPPER_BODY_WIDTH_FACE_RATIO  4.8f
#define CAPTURE_UPPER_BODY_HEIGHT_FACE_RATIO 6.4f
#define CAPTURE_UPPER_BODY_MIN_WIDTH_RATIO   0.82f
#define CAPTURE_UPPER_BODY_MIN_HEIGHT_RATIO  0.88f
#define CAPTURE_UPPER_BODY_CENTER_Y_RATIO    1.58f
#define CAPTURE_UPPER_BODY_TOP_DIVISOR       3.4f
// 评分惩罚：运动、遮挡、fallback 路径、模糊都会拉低候选得分。
#define CAPTURE_MOTION_SCORE_PENALTY    300.0f
#define CAPTURE_OCCLUSION_SCORE_PENALTY 260.0f
#define CAPTURE_FALLBACK_SCORE_PENALTY  140.0f
#define CAPTURE_MAX_BLUR_SEVERITY       0.62f
#define CAPTURE_FALLBACK_MAX_BLUR_SEVERITY 0.78f
#define CAPTURE_BLUR_SEVERITY_SCORE_PENALTY 380.0f
// 低照度自适应防模糊：亮度低于 threshold 后，逐步收紧运动/模糊/清晰度门槛；到 floor 时收紧到最严格。
#define CAPTURE_LOW_LIGHT_BRIGHTNESS_THRESHOLD 92.0
// Low-light anti-blur tuning: tighten motion/blur/clarity gates when the scene gets dark.
#define CAPTURE_LOW_LIGHT_BRIGHTNESS_THRESHOLD 92.0
#define CAPTURE_LOW_LIGHT_BRIGHTNESS_FLOOR     58.0
#define CAPTURE_LOW_LIGHT_MOTION_RATIO_SCALE   0.72f
#define CAPTURE_LOW_LIGHT_MOTION_REJECT_RATIO_SCALE 0.60f
#define CAPTURE_LOW_LIGHT_MAX_BLUR_SEVERITY_SCALE 0.82f
#define CAPTURE_LOW_LIGHT_FALLBACK_MAX_BLUR_SEVERITY_SCALE 0.74f
#define CAPTURE_LOW_LIGHT_MIN_CLARITY_SCALE    1.10f
#define CAPTURE_LOW_LIGHT_FALLBACK_MIN_CLARITY_SCALE 1.18f

// 亮度与 IR-CUT 配置
// 每隔 N 帧采样一次场景亮度，避免每帧测光带来额外开销。
#define CAMERA_BRIGHTNESS_SAMPLE_INTERVAL  5
// 高于白片阈值切回白天模式，低于黑片阈值切到夜视模式。
#define CAMERA_BRIGHTNESS_WHITE_THRESHOLD  110.0
#define CAMERA_BRIGHTNESS_BLACK_THRESHOLD   85.0

// 软件亮度补偿参数（曝光不足时自动增益）
// 目标亮度：只在画面偏暗时向该值靠拢，不等于强制拉到这个均值。
#define CAMERA_BRIGHTNESS_TARGET           105.0   // 目标亮度均值
// 触发阈值：低于该值才启用软件提亮。
#define CAMERA_BRIGHTNESS_BOOST_THRESHOLD   80.0   // 低于此值启动软件增益
// 过暗地板：低于该值说明已接近纯噪声，继续提亮收益很低。
#define CAMERA_BRIGHTNESS_BOOST_MIN_FLOOR   25.0   // 低于此值不做补偿（全噪声无意义）
// 线性补偿上限：防止过度放大噪声和高光溢出。
#define CAMERA_BRIGHTNESS_MAX_ALPHA          2.2    // 最大乘性增益（防止过曝/噪声放大）
#define CAMERA_BRIGHTNESS_MAX_BETA          25.0    // 最大加性偏移（暗部抬升）
// Gamma < 1 时优先抬暗部、尽量保住亮部层次。
#define CAMERA_BRIGHTNESS_GAMMA              0.85   // Gamma 校正指数（<1 提亮暗部、保留亮部）
// 极暗场景下不追求满目标亮度，而是按比例收缩目标，抑制噪声放大。
#define CAMERA_BRIGHTNESS_DARK_BLEND         0.55   // 极暗场景(亮度<55)目标缩放因子
