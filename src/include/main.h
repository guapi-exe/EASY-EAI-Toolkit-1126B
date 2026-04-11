#pragma once

// 常用数组长度宏。
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

// 相机原始采集分辨率。
#define CAMERA_WIDTH    2688
#define CAMERA_HEIGHT   1520
// 算法处理使用的缩放分辨率。
#define IMAGE_WIDTH     1280
#define IMAGE_HEIGHT    720
// 板端两个摄像头节点编号。
#define CAMERA_INDEX_1  11
#define CAMERA_INDEX_2  51
// 默认上报的摄像头编号。
#define DEFAULT_CAMERA_NUMBER 1
// 相机输出像素格式。
#define CAMERA_FORMAT   RK_FORMAT_BGR_888
// BGR 三通道字节数。
#define IMGRATIO        3
// 单帧图像总字节数。
#define IMAGE_SIZE      (CAMERA_WIDTH * CAMERA_HEIGHT * IMGRATIO)
// 跟踪目标允许连续丢失的最大帧数。
#define MAX_MISSED      45

// 运行时加载的模型文件名。
#define PERSON_MODEL_PATH   "person_detect.model"
#define FACE_MODEL_PATH     "face_detect.model"
#define NAFNET_TINY_MODEL_PATH "nafnet_tiny.rknn"

// RetinaFace 测试程序使用的模型和输入参数。
#define RETIAN_MODEL_TYPE   0
#define RETIAN_INPUT_H      480
#define RETIAN_INPUT_W      640
#define RETIAN_CONF_THRESH  0.7f
#define RETIAN_NMS_THRESH   0.4f


// 抓拍候选与评分阈值。
#define CAPTURE_MIN_CLARITY              110.0   // 强候选最低清晰度
#define CAPTURE_FALLBACK_MIN_CLARITY     55.0    // 弱候选最低清晰度
#define CAPTURE_MAX_MOTION_RATIO         0.014f  // 评分阶段允许的最大运动比例
#define CAPTURE_MAX_MOTION_REJECT_RATIO  0.038f  // 硬拒绝阶段允许的最大运动比例

// 检测与候选评估调度参数。
#define CAPTURE_PERSON_DETECT_INTERVAL   2       // 每隔多少帧跑一次人体检测
#define CAPTURE_FACE_DETECT_INTERVAL     2       // 每隔多少帧跑一次人脸检测
#define CAPTURE_FACE_INPUT_MAX_WIDTH     640     // 人脸检测输入的最大宽度
#define CAPTURE_MAX_FRAME_CANDIDATES     48      // 每个 track 保留的候选帧上限
#define CAPTURE_CANDIDATE_QUEUE_MAX      128     // 候选评估队列上限
#define CAPTURE_CANDIDATE_PER_TRACK_MAX_PENDING 6 // 单个 track 允许排队的候选任务数

// 人体 ROI 扩展比例，用于从整帧裁出更完整的人体区域。
#define CAPTURE_PERSON_CONTEXT_EXPAND_X  0.12f   // 左右扩边比例
#define CAPTURE_PERSON_CONTEXT_EXPAND_TOP 0.18f  // 上方扩边比例
#define CAPTURE_PERSON_CONTEXT_EXPAND_BOTTOM 0.10f // 下方扩边比例

// 人脸候选基础筛选阈值。
#define CAPTURE_MIN_FACE_SCORE           0.55f   // 人脸检测置信度下限
#define CAPTURE_HEADSHOT_EXPAND_RATIO    1.80f   // 头像裁剪框相对人脸框的放大倍数
#define CAPTURE_HEADSHOT_DOWN_SHIFT      0.20f   // 头像裁剪中心向下偏移比例
#define CAPTURE_FOCUS_SCALE_FACTOR       3       // 清晰度计算的降采样倍率
#define CAPTURE_MIN_AREA_RATIO           0.025f  // 最低人体面积占比，小于此值直接判远
#define CAPTURE_NEAR_AREA_RATIO          0.065f  // 认为目标已经靠近的面积占比
#define CAPTURE_AREA_SCORE_TARGET_RATIO  0.6f    // 面积评分的理想目标占比

// 接近趋势与 track 稳定性判断。
#define CAPTURE_APPROACH_RATIO_POS       0.16f   // 面积增长超过该比例视为接近
#define CAPTURE_APPROACH_RATIO_NEG      -0.16f   // 面积下降超过该比例视为远离
#define CAPTURE_MIN_TRACK_HITS           3       // track 至少命中多少次才参与抓拍
#define CAPTURE_REQUIRE_APPROACH         1       // 是否要求目标处于接近状态
#define CAPTURE_REQUIRE_FRONTAL_FACE     1       // 是否要求正脸优先

// 人脸姿态与几何约束。
#define CAPTURE_MAX_YAW                  0.8f    // 强候选最大偏航容忍
#define CAPTURE_FALLBACK_MAX_YAW         0.95f   // 弱候选最大偏航容忍
#define CAPTURE_STRONG_FRONTAL_MAX_ROLL  18.0f   // 强正脸最大翻滚角
#define CAPTURE_STRONG_FRONTAL_MAX_YAW   0.22f   // 强正脸最大偏航角
#define CAPTURE_FACE_MIN_AREA_IN_PERSON  0.018f  // 人脸面积在人形框中的最小占比
#define CAPTURE_FACE_MIN_WIDTH_RATIO     0.085f  // 人脸宽度在人形框中的最小占比
#define CAPTURE_FACE_MIN_CENTER_Y_RATIO  0.10f   // 人脸中心 Y 的最小相对位置
#define CAPTURE_FACE_MAX_CENTER_Y_RATIO  0.58f   // 人脸中心 Y 的最大相对位置
#define CAPTURE_MIN_FACE_BOX_SHORT_SIDE  76      // 人脸框短边最小像素值
#define CAPTURE_MIN_FACE_BOX_AREA        9000    // 人脸框最小面积

// 遮挡与头像边缘留白容忍度。
// 遮挡容忍度：多人场景中遮挡不可避免，适度放宽减少漏抓
#define CAPTURE_MAX_PERSON_OCCLUSION     0.55f   // 人体遮挡上限
#define CAPTURE_MAX_FACE_EDGE_OCCLUSION  0.65f   // 人脸贴边遮挡上限
#define CAPTURE_FACE_EDGE_MIN_MARGIN     0.08f   // 判断人脸贴边的最小边距比例
#define CAPTURE_HEADSHOT_MIN_FACE_MARGIN 0.08f   // 强候选头像裁剪要求的人脸边距
#define CAPTURE_FALLBACK_HEADSHOT_MIN_FACE_MARGIN 0.01f // 弱候选头像裁剪最小边距
#define CAPTURE_FALLBACK_MAX_FACE_EDGE_OCCLUSION 0.92f // 弱候选允许的人脸贴边程度

// 上半身图裁剪参数。
#define CAPTURE_UPPER_BODY_WIDTH_FACE_RATIO   4.8f  // 上半身框宽度相对人脸宽度倍数
#define CAPTURE_UPPER_BODY_HEIGHT_FACE_RATIO  6.4f  // 上半身框高度相对人脸高度倍数
#define CAPTURE_UPPER_BODY_MIN_WIDTH_RATIO    0.82f // 上半身框相对人体框的最小宽度
#define CAPTURE_UPPER_BODY_MIN_HEIGHT_RATIO   0.88f // 上半身框相对人体框的最小高度
#define CAPTURE_UPPER_BODY_CENTER_Y_RATIO     1.58f // 上半身框中心相对人脸中心的 Y 比例
#define CAPTURE_UPPER_BODY_TOP_DIVISOR        3.4f  // 上半身框顶部回拉系数

// 评分惩罚项权重。
#define CAPTURE_MOTION_SCORE_PENALTY     300.0f  // 运动带来的评分惩罚
#define CAPTURE_OCCLUSION_SCORE_PENALTY  260.0f  // 遮挡带来的评分惩罚
#define CAPTURE_FALLBACK_SCORE_PENALTY   140.0f  // 弱候选额外惩罚
#define CAPTURE_MAX_BLUR_SEVERITY        0.62f   // 强候选最大模糊度
#define CAPTURE_FALLBACK_MAX_BLUR_SEVERITY 0.78f // 弱候选最大模糊度
#define CAPTURE_BLUR_SEVERITY_SCORE_PENALTY 380.0f // 模糊度评分惩罚权重

// 环境亮度采样与 IR-CUT 切换阈值。
#define CAMERA_BRIGHTNESS_SAMPLE_INTERVAL  5      // 每隔多少帧采样一次环境亮度
#define CAMERA_BRIGHTNESS_WHITE_THRESHOLD  110.0  // 切回白天模式的亮度阈值
#define CAMERA_BRIGHTNESS_BLACK_THRESHOLD   85.0  // 切到夜视模式的亮度阈值

// 软件提亮参数，用于暗光场景补偿。
#define CAMERA_BRIGHTNESS_TARGET           105.0  // 提亮后的目标平均亮度
#define CAMERA_BRIGHTNESS_BOOST_THRESHOLD   80.0  // 低于该亮度开始启用提亮
#define CAMERA_BRIGHTNESS_BOOST_MIN_FLOOR   25.0  // 极暗场景保护下限
#define CAMERA_BRIGHTNESS_MAX_ALPHA          2.2   // 线性增益上限
#define CAMERA_BRIGHTNESS_MAX_BETA          25.0   // 亮度偏移上限
#define CAMERA_BRIGHTNESS_GAMMA              0.85  // Gamma 校正参数
#define CAMERA_BRIGHTNESS_DARK_BLEND         0.55  // 暗部融合权重
