# Face Detect SCRFD

SCRFD (Sample and Computation Redistribution Face Detector) 人脸检测算法的 RKNN 实现。

## 模型特点

- **Anchor-Free**: 使用中心点回归，无需预定义 anchor
- **多尺度检测**: 3 个 stride (8/16/32) 覆盖不同尺寸人脸
- **高精度**: 相比 RetinaFace 误检率显著降低，尤其是背头/侧脸/假脸
- **无 Landmark**: 标准版本仅输出 bbox + score，专注检测准确性

## 模型输入输出

### 输入
- **norm_tensor:0**: `[1, 3, 640, 640]` - BGR 图像，归一化 (pixel-127.5)/128

### 输出 (SCRFD-1G)
| 输出编号 | 名称 | 维度 | 说明 |
|---------|------|------|------|
| output  | norm_tensor:1 | `[1,12800,1]` | stride 8 置信度 |
| output1 | norm_tensor:2 | `[1,3200,1]`  | stride 16 置信度 |
| output2 | norm_tensor:3 | `[1,800,1]`   | stride 32 置信度 |
| output3 | norm_tensor:4 | `[1,12800,4]` | stride 8 bbox (l,t,r,b) |
| output4 | norm_tensor:5 | `[1,3200,4]`  | stride 16 bbox (l,t,r,b) |
| output5 | norm_tensor:6 | `[1,800,4]`   | stride 32 bbox (l,t,r,b) |

### 特征图尺寸
| stride | 特征图尺寸 | anchor 数量 |
|--------|-----------|------------|
| 8      | 80×80     | 12800      |
| 16     | 40×40     | 3200       |
| 32     | 20×20     | 800        |

## API 使用

### 1. 初始化
```cpp
#include "face_detect_scrfd.h"

rknn_context ctx = 0;
SCRFDConfig config = get_scrfd_config(640, 640);
config.conf_thresh = 0.65f;  // 推荐阈值
config.nms_thresh = 0.4f;

int ret = face_detect_scrfd_init(&ctx, "scrfd_1g_shape640x640.rknn", &config);
```

### 2. 推理
```cpp
cv::Mat image = cv::imread("test.jpg");
std::vector<SCRFDResult> results;

int num_faces = face_detect_scrfd_run(ctx, image, results);

for (auto &face : results) {
    printf("Face at (%.0f,%.0f,%.0f,%.0f), score: %.3f\n",
           face.box.x, face.box.y, 
           face.box.br().x, face.box.br().y, 
           face.score);
}
```

### 3. 释放
```cpp
face_detect_scrfd_release(ctx);
```

## 配置参数

### SCRFDConfig 结构体
```cpp
struct SCRFDConfig {
    int input_height;          // 模型输入高度 (640)
    int input_width;           // 模型输入宽度 (640)
    std::vector<int> strides;  // 特征金字塔步长 {8,16,32}
    float conf_thresh;         // 置信度阈值 (0.6~0.75)
    float nms_thresh;          // NMS 阈值 (0.4~0.45)
};
```

### 推荐阈值 (RK1126B)
| 场景 | conf_thresh | nms_thresh | 说明 |
|-----|------------|-----------|------|
| 通用 | 0.65 | 0.4 | 平衡精度和召回 |
| 高精度 | 0.75 | 0.4 | 减少误检 |
| 高召回 | 0.5 | 0.45 | 检测更多人脸 |

## 关键差异: SCRFD vs RetinaFace

| 特性 | SCRFD | RetinaFace |
|------|-------|-----------|
| 检测方式 | Anchor-Free | Anchor-Based |
| 输出格式 | 单通道 sigmoid | 2 通道 softmax |
| bbox 格式 | ltrb 偏移 | center + wh |
| Landmark | 无 (标准版) | 5 点 |
| 背头误检 | 低 ✅ | 高 ❌ |
| 侧脸准确性 | 高 ✅ | 中等 |

## 解码流程

### 1. Score 解码
```cpp
// SCRFD 使用单通道 sigmoid
float score = sigmoid(raw_score[i]);
if (score < conf_thresh) continue;
```

### 2. BBox 解码 (Anchor-Free)
```cpp
// 获取 ltrb 偏移
float l = bbox[0];  // 左偏移
float t = bbox[1];  // 上偏移
float r = bbox[2];  // 右偏移
float b = bbox[3];  // 下偏移

// 计算中心点
float cx = (x + 0.5) * stride;
float cy = (y + 0.5) * stride;

// 解码为绝对坐标
float x1 = cx - l * stride;
float y1 = cy - t * stride;
float x2 = cx + r * stride;
float y2 = cy + b * stride;
```

### 3. NMS 后处理
- 按置信度降序排列
- IoU > nms_thresh 的框被抑制
- 保留最高分框

## 性能对比

### RK1126B NPU (640×640 输入)
| 模型 | 速度 | 背头误检 | 侧脸检测 | 推荐 |
|-----|------|---------|---------|------|
| RetinaFace-0.25 | 快 | 高 ❌ | 中 | ⚠️ |
| SCRFD-0.5G | 中 | 低 ✅ | 高 | ✅ |
| SCRFD-1G | 稍慢 | 极低 ✅ | 极高 | ⭐⭐⭐ |
| SCRFD-2.5G | 中慢 | 极低 ✅ | 极高 | ⭐⭐⭐ |

## 典型问题

### Q1: 背头仍被误检？
- 提高 `conf_thresh` 至 0.7~0.75
- 确认模型是 SCRFD，不是 RetinaFace
- 检查输入图像是否模糊

### Q2: 小脸检测不到？
- 降低 `conf_thresh` 至 0.5~0.6
- 使用 SCRFD-2.5G 模型 (更强)
- 确认人脸像素 > 20×20

### Q3: 速度太慢？
- 使用 SCRFD-0.5G 模型
- 降低输入分辨率至 320×320
- 开启 RKNN 优化选项

## 文件结构
```
face_detect_scrfd/
├── face_detect_scrfd.h       # 头文件
├── face_detect_scrfd.cpp     # 实现文件
├── postprocess.h             # 后处理通用头
├── rknn_api.h                # RKNN SDK 头文件
├── rknn_matmul_api.h         # RKNN 矩阵运算头文件
├── api.cmake                 # CMake 配置
└── README.md                 # 本文档
```

## 依赖
- RKNN SDK (RV1109/RV1126/RK3566/RK3568/RK3588)
- OpenCV 4.x
- C++11 或更高

## 作者
AI Assistant

## 许可证
参考项目主许可证
