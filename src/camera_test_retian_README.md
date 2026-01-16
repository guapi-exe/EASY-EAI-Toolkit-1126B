# Camera Test RetinaFace - 人脸检测测试程序

基于 `camera_test.cpp` 创建的 RetinaFace 人脸检测实时测试程序。

## 功能特性

- ✅ 实时摄像头采集（4K）
- ✅ RetinaFace 人脸检测（720p）
- ✅ 人脸边界框和5个关键点可视化
- ✅ FPS 统计
- ✅ 检测人脸数统计
- ✅ 性能分析（捕获、缩放、检测耗时）
- ✅ 可选保存检测结果图片
- ✅ 支持三种模型配置

## 编译

```bash
cd src
chmod +x build_camera_test_retian.sh
./build_camera_test_retian.sh
```

编译成功后，可执行文件位于 `build/camera_test_retian`

## 使用方法

### 基本用法

```bash
cd build
./camera_test_retian --model ./retinaface_480x640.rknn
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model <path>` | .rknn 模型文件路径 | `./retinaface_480x640.rknn` |
| `--type <0\|1\|2>` | 模型类型：0=RETINAFACE, 1=SLIM, 2=RFB | 0 |
| `--save` | 保存检测结果图片 | 否 |
| `--interval N` | 保存间隔帧数 | 50 |
| `--conf <0.0-1.0>` | 置信度阈值 | 0.5 |
| `--nms <0.0-1.0>` | NMS 阈值 | 0.4 |
| `--help` | 显示帮助信息 | - |

### 使用示例

#### 1. 基本检测（RETINAFACE 模型）
```bash
./camera_test_retian --model ./retinaface_480x640.rknn
```

#### 2. 使用 RFB 模型，保存检测结果
```bash
./camera_test_retian --model ./rfb_480x640.rknn --type 2 --save
```

#### 3. 调整检测阈值，提高精度
```bash
./camera_test_retian --model ./retinaface_480x640.rknn --conf 0.6 --nms 0.3
```

#### 4. 快速保存检测结果（每20帧）
```bash
./camera_test_retian --save --interval 20
```

## 输出说明

### 运行时输出

```
=== RetinaFace 人脸检测测试 ===
摄像头分辨率: 3840x2160 (2K)
模型路径: ./retinaface_480x640.rknn
模型类型: RETINAFACE
置信度阈值: 0.50
NMS阈值: 0.40
摄像头初始化成功
Generated 16800 prior boxes
RetinaFace model initialized successfully
RetinaFace 模型初始化成功
开始检测，按 Ctrl+C 停止...
=================================
帧统计: 总帧数=30, FPS=15.0, 检测人脸=2(平均1.8), 运行时间=2s
帧统计: 总帧数=60, FPS=15.2, 检测人脸=1(平均1.5), 运行时间=4s
...
```

### 最终统计

```
=================================
测试结束统计:
  总帧数: 450
  运行时间: 30.12 秒
  平均FPS: 14.94
  总检测人脸数: 723
  平均每帧人脸数: 1.61
性能分析:
  平均帧捕获耗时: 2.34 ms
  平均缩放耗时: 18.45 ms
  平均检测耗时: 42.67 ms
  平均绘制耗时: 5.23 ms
=================================
```

### 保存的图片

如果启用了 `--save` 选项，会保存以下格式的图片：

```
retian_frame_000050_faces_2.jpg  # 第50帧，检测到2张人脸
retian_frame_000100_faces_1.jpg  # 第100帧，检测到1张人脸
...
```

图片包含：
- 🔵 **蓝色/绿色/红色边界框** - 人脸位置
- 🟡 **黄色圆点** - 左眼、右眼
- 🟣 **品红圆点** - 鼻尖
- 🟢 **绿色圆点** - 左嘴角、右嘴角
- 📝 **白色文字** - 置信度百分比

## 性能优化建议

### 1. 提高 FPS
- 使用更小的输入分辨率
- 降低置信度阈值（减少后处理时间）
- 关闭图片保存

### 2. 提高检测精度
- 提高置信度阈值（`--conf 0.6`）
- 调整 NMS 阈值（`--nms 0.3`）
- 使用原始 RETINAFACE 模型（非 SLIM/RFB）

### 3. 平衡速度和精度
- 使用 RFB 模型 (`--type 2`)
- 置信度 0.5-0.6
- NMS 阈值 0.4

## 常见问题

### Q: 检测不到人脸？
A: 
1. 检查模型路径是否正确
2. 降低置信度阈值 `--conf 0.3`
3. 确认摄像头画面中有人脸
4. 检查光照条件

### Q: FPS 太低？
A:
1. 检测耗时是主要瓶颈（40-50ms）
2. 可以考虑使用 SLIM 模型（`--type 1`）
3. 或降低检测频率（每N帧检测一次）

### Q: 保存的图片在哪里？
A: 在程序运行目录（`build/`）下，文件名格式 `retian_frame_XXXXXX_faces_N.jpg`

### Q: 如何退出程序？
A: 按 `Ctrl+C` 优雅退出，会显示完整统计信息

## 对比测试

与现有 `face_detect` 模块对比：

| 特性 | face_detect | face_detect_retian |
|------|-------------|-------------------|
| 检测框 | ✅ | ✅ |
| 关键点 | ✅ (5点) | ✅ (5点) |
| 模型类型 | 单一 | 3种可选 |
| Prior boxes | 预生成 | 动态生成 |
| NMS | 内置 | 自定义 |
| 性能 | 快 | 中等 |

## 技术细节

- **摄像头**: MIPI 4K (3840x2160)
- **处理分辨率**: 720p (1280x720)
- **模型输入**: 480x640
- **检测输出**: 边界框 + 5个关键点
- **关键点顺序**: 左眼、右眼、鼻尖、左嘴角、右嘴角

## 相关文件

- [camera_test_retian.cpp](camera_test_retian.cpp) - 测试程序源码
- [face_detect_retian.h](../easyeai-api/algorithm/face_detect_retian/face_detect_retian.h) - RetinaFace 接口
- [face_detect_retian.cpp](../easyeai-api/algorithm/face_detect_retian/face_detect_retian.cpp) - RetinaFace 实现
- [CMakeLists.txt](CMakeLists.txt) - CMake 配置
