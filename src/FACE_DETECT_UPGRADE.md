# 人脸检测模型升级说明

## 修改概要

已将人脸检测模块从旧的 `face_detect` 升级为新的 `face_detect_retian` (RetinaFace)。

## 主要修改

### 1. main.h - 添加模型配置
```cpp
// 模型路径配置
#define PERSON_MODEL_PATH   "person_detect.model"
#define FACE_MODEL_PATH     "retinaface_480x640.rknn"

// RetinaFace 配置
#define RETIAN_MODEL_TYPE   0  // 0=RETINAFACE, 1=SLIM, 2=RFB
#define RETIAN_INPUT_H      480
#define RETIAN_INPUT_W      640
#define RETIAN_CONF_THRESH  0.5f
#define RETIAN_NMS_THRESH   0.4f
```

**优点**：
- 集中管理模型配置
- 便于修改模型路径和参数
- 无需修改代码即可调整检测阈值

### 2. camera_task.cpp - 替换为 RetinaFace

#### 头文件
```cpp
#include "face_detect_retian.h"  // 替换 face_detect.h
```

#### 初始化
```cpp
RetinaFaceConfig faceConfig = get_retian_config(
    (RetinaFaceModelType)RETIAN_MODEL_TYPE, 
    RETIAN_INPUT_H, 
    RETIAN_INPUT_W
);
face_detect_retian_init(&faceCtx, faceModelPath.c_str(), &faceConfig);
```

#### 检测
```cpp
std::vector<RetinaFaceResult> face_result;
int num_faces = face_detect_retian_run(faceCtx, person_roi_resized, face_result, 
                                       RETIAN_CONF_THRESH, RETIAN_NMS_THRESH, 10);
```

#### 结果处理
- `RetinaFaceResult` 使用 `cv::Rect_<float>` 和 `std::vector<cv::Point2f>`
- 需要显式转换为整数类型的 Rect

### 3. main.cpp - 使用配置宏
```cpp
CameraTask camera(PERSON_MODEL_PATH, FACE_MODEL_PATH, CAMERA_INDEX_1);
```

### 4. CMakeLists.txt - 更新编译配置
- 添加 `face_detect_retian_srcs` 源文件
- main 目标链接 RetinaFace 源文件和库

## 新旧模型对比

| 特性 | 旧 face_detect | 新 face_detect_retian |
|------|---------------|----------------------|
| 检测框 | ✅ | ✅ |
| 关键点 | ✅ (5点) | ✅ (5点) |
| 返回类型 | `std::vector<det>` | `std::vector<RetinaFaceResult>` |
| 坐标类型 | `cv::Rect_<float>` | `cv::Rect_<float>` |
| 配置灵活性 | 固定 | 3种模型可选 |
| 阈值控制 | 固定 | 可配置 |

## 使用方法

### 快速开始
默认配置已在 main.h 中设置，直接编译运行即可。

### 修改模型路径
编辑 [main.h](include/main.h)：
```cpp
#define FACE_MODEL_PATH "your_model_path.rknn"
```

### 调整检测参数

#### 1. 更换模型类型
```cpp
#define RETIAN_MODEL_TYPE 2  // 使用 RFB 模型
```

#### 2. 调整置信度阈值
```cpp
#define RETIAN_CONF_THRESH 0.6f  // 提高精度，减少误检
```

#### 3. 调整 NMS 阈值
```cpp
#define RETIAN_NMS_THRESH 0.3f  // 更严格的重叠过滤
```

#### 4. 修改输入尺寸（需匹配模型）
```cpp
#define RETIAN_INPUT_H 640
#define RETIAN_INPUT_W 640
```

## 编译

```bash
cd build
cmake ../src
make main -j$(nproc)
```

## 注意事项

1. **模型文件**：确保 `retinaface_480x640.rknn` 存在于运行目录
2. **坐标转换**：RetinaFace 返回的坐标需要缩放到原始图像尺寸
3. **关键点顺序**：左眼、右眼、鼻尖、左嘴角、右嘴角
4. **性能**：RetinaFace 比原模型稍慢，但精度更高

## 兼容性

- ✅ 保持原有的人脸姿态判断逻辑 (`isFrontalFace`, `isSideFace`)
- ✅ 保持原有的清晰度评分系统
- ✅ 保持原有的人脸框扩展逻辑
- ✅ 关键点绘制坐标已修正

## 测试程序

独立的 RetinaFace 测试程序：
```bash
./camera_test_retian --model ./retinaface_480x640.rknn --save
```

详见：[camera_test_retian_README.md](camera_test_retian_README.md)
