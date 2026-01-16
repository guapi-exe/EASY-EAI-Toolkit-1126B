# RetinaFace C++ Implementation for RKNN

本模块是基于 Python 版本的 RetinaFace 人脸检测模型移植到 C++ 的 RKNN 实现。

## 功能特性

- 支持 RetinaFace、Slim、RFB 三种模型配置
- 输出人脸边界框和 5 个关键点（双眼、鼻尖、嘴角）
- NMS 后处理
- 自动缩放回原图尺寸
- 高性能 NPU 推理

## 文件说明

- `face_detect_retian.h` - 头文件，定义接口和数据结构
- `face_detect_retian.cpp` - 实现文件，包含模型加载、推理、后处理
- `postprocess.h` - 通用后处理结构定义
- `test_retian.cpp` - 示例测试程序
- `api.cmake` - CMake 配置文件
- `rknn_api.h` / `rknn_matmul_api.h` - RKNN SDK 头文件

## 使用方法

### 1. 初始化模型

```cpp
#include "face_detect_retian.h"

// 获取预定义配置（输入尺寸 480x640）
RetinaFaceConfig config = get_retian_config(RETINAFACE_MODEL, 480, 640);

// 初始化模型
rknn_context ctx;
int ret = face_detect_retian_init(&ctx, "retinaface_480x640.rknn", &config);
if (ret != 0) {
    printf("Failed to initialize model\n");
    return -1;
}
```

### 2. 执行推理

```cpp
cv::Mat image = cv::imread("test.jpg");
std::vector<RetinaFaceResult> results;

// 运行检测
int num_faces = face_detect_retian_run(
    ctx,            // RKNN context
    image,          // 输入图像 (BGR)
    results,        // 输出结果
    0.5f,          // 置信度阈值
    0.4f,          // NMS 阈值
    10             // 最多保留人脸数
);

// 处理结果
for (int i = 0; i < num_faces; i++) {
    printf("Face %d: score=%.3f\n", i, results[i].score);
    printf("  Box: (%.0f, %.0f, %.0f, %.0f)\n",
           results[i].box.x, results[i].box.y,
           results[i].box.br().x, results[i].box.br().y);
    
    // 5个关键点：左眼、右眼、鼻尖、左嘴角、右嘴角
    for (int j = 0; j < 5; j++) {
        printf("  Landmark %d: (%.0f, %.0f)\n", j,
               results[i].landmarks[j].x, results[i].landmarks[j].y);
    }
}
```

### 3. 释放资源

```cpp
face_detect_retian_release(ctx);
```

## 模型配置

支持三种预定义配置：

| 类型 | 枚举值 | min_sizes | steps | 适用场景 |
|------|--------|-----------|-------|---------|
| RetinaFace | `RETINAFACE_MODEL` | [[10,20], [32,64], [128,256]] | [8,16,32] | 标准人脸检测 |
| Slim | `SLIM_MODEL` | [[10,16,24], [32,48], [64,96], [128,192,256]] | [8,16,32,64] | 轻量级模型 |
| RFB | `RFB_MODEL` | [[10,16,24], [32,48], [64,96], [128,192,256]] | [8,16,32,64] | RFB优化版本 |

## 数据结构

### RetinaFaceResult

```cpp
class RetinaFaceResult {
public:
    cv::Rect_<float> box;              // 人脸边界框
    std::vector<cv::Point2f> landmarks; // 5个关键点
    float score;                        // 置信度
    
    void print();  // 打印结果
};
```

### RetinaFaceConfig

```cpp
struct RetinaFaceConfig {
    RetinaFaceModelType model_type;    // 模型类型
    std::vector<std::vector<int>> min_sizes;  // anchor 最小尺寸
    std::vector<int> steps;            // 特征图步长
    float variance[2];                 // 解码方差 [0.1, 0.2]
    int input_height;                  // 模型输入高度
    int input_width;                   // 模型输入宽度
};
```

## 编译示例

### 使用 CMake

```cmake
include(easyeai-api/algorithm/face_detect_retian/api.cmake)

add_executable(test_retian 
    easyeai-api/algorithm/face_detect_retian/test_retian.cpp
    easyeai-api/algorithm/face_detect_retian/face_detect_retian.cpp
)

target_include_directories(test_retian PRIVATE 
    ${FACE_DETECT_RETIAN_INCLUDE_DIRS}
)

target_link_directories(test_retian PRIVATE 
    ${FACE_DETECT_RETIAN_LIBS_DIRS}
)

target_link_libraries(test_retian 
    ${FACE_DETECT_RETIAN_LIBS}
)
```

### 运行测试程序

```bash
./test_retian ./retinaface_480x640.rknn test.jpg 0
```

参数说明：
- 第一个参数：.rknn 模型路径
- 第二个参数：测试图片路径
- 第三个参数：模型类型 (0=RETINAFACE, 1=SLIM, 2=RFB)

## 性能参数

- **输入格式**: BGR, NHWC
- **归一化**: (pixel - 127.5) / 128.0
- **输出数量**: 3 个 (loc, conf, landms)
- **后处理**: 包含 bbox 解码、landmark 解码、NMS

## 注意事项

1. 模型输入尺寸需与配置一致（例如 480x640）
2. 输入图像会自动缩放到模型尺寸，输出结果会缩放回原图尺寸
3. 置信度阈值建议 0.5，NMS 阈值建议 0.4
4. 关键点顺序：左眼、右眼、鼻尖、左嘴角、右嘴角

## 参考

- Python 实现: `face-detector-rknn-main/model/face_detector.py`
- YOLOv5 示例: `yolov5_detect_C_demo`
- 现有 face_detect: `easyeai-api/algorithm/face_detect`
