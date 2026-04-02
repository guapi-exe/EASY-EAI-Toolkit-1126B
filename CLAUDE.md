# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个基于 RK3588 平台的智能相机边缘设备项目,实现实时人脸/人体检测、抓拍筛选、图像上传和 TCP 远程控制功能。

## 构建与部署

### 本地构建
```bash
cd src
./build.sh
```
- 产物: `Release/main`
- 如果设置了 `$SYSROOT` 环境变量,会自动复制到板端目录 `$SYSROOT/userdata/Demo/`

### 清理构建
```bash
cd src
./build.sh clear
```

### 板端更新
```bash
cd src
./update.sh
```
- 执行 `git fetch && git reset --hard origin/master` 后重新构建

### 交叉编译
- 非 armv7l/aarch64 主机会自动引入 `$HOME/configs/cross.cmake`
- CMake 会检测 GCC 12 aarch64 编译器 ICE 问题并自动降级优化等级

### 验证程序
默认编译 `camera_test_advanced`,其他测试程序需手动启用:
- `camera_test.cpp` - 基础相机测试
- `camera_test_retian.cpp` - RetinaFace 模型测试
- `camera_test_scrfd.cpp` - SCRFD 模型测试  
- `camera_streaming.cpp` - RTMP/RTSP 推流测试

## 核心架构

### 主程序入口 (src/main.cpp)

启动时创建三个核心任务并注册回调:

1. **CameraTask** - 相机采集与抓拍筛选
2. **UploaderTask** - 图像队列化上传
3. **TcpClient** - 设备协议与远程控制

**调试模式**:
```bash
./main debug
```
- 上传路径切换到 `/receive/image/auto` (旧接口)
- 正常模式使用 `/receive/image/auto/minio`

### CameraTask (src/utils/tasks/camera_task.cpp)

**职责**:
- MIPI 相机取帧 (通过 `src/commonApi/camera.h`)
- 环境亮度采样与软件提亮 (brightness_boost)
- IR-CUT 自动切换 (GPIO 184/185, sysfs 控制)
- 人脸/人体检测 (RKNN 推理)
- SORT 多目标跟踪与轨迹评估
- 候选图像筛选 (清晰度、模糊度、遮挡、接近趋势)

**关键阈值** (可通过 `device_config.json` 调整):
- `CAPTURE_MIN_CLARITY` - 最低清晰度
- `CAPTURE_MAX_BLUR_SEVERITY` - 最大模糊度
- `CAPTURE_MAX_MOTION_RATIO` - 最大运动比例
- IR-CUT 切换间隔: 60 秒,切换后稳定期 8 秒

**暗光自适应**:
- 亮度低于 92.0 时触发放宽策略
- 动态调整清晰度/模糊度/运动阈值
- 多帧融合增强 (实验性)

### UploaderTask (src/utils/tasks/uploader_task.cpp)

**职责**:
- libcurl HTTP multipart 上传
- JPEG 压缩 (目标 ~300KB)
- 长边缩放与轻量增强
- person/face 图像配对 (通过 uniqueCode)

**上传类型**:
- `person` - 人体全身图
- `face` - 人脸特写图
- `manual` - 手动抓拍 (独立路径)

### TcpClient (src/utils/tcp_client.cpp)

**协议**: JSON line (详见 `src/TCP_PROTOCOL.md`)

**设备 → 服务器**:
- `register` - 设备注册
- `heartbeat` - 心跳 (含 CPU/内存/环境亮度)
- `person_immediate` - 人员出现事件
- `all_person_left` - 所有人员离开
- `capture_complete` - 抓拍完成

**服务器 → 设备**:
- `sleep` - 进入低功耗模式
- `wake` - 恢复采集
- `capture` - 立即抓拍
- `config_update` - 更新配置并回写 `device_config.json`

### DeviceConfig (src/utils/device_config.cpp)

**配置文件**: `device_config.json` (运行时自动创建)

**关键字段**:
- `device.code` - 设备编号
- `upload.server` - 上传服务器地址
- `tcp.server_ip` / `tcp.port` - TCP 服务器
- `capture_defaults` - 抓拍阈值
- `ircut.brightness_black_threshold` - IR-CUT 切换阈值
- `brightness_boost` - 软件提亮参数

配置可通过 TCP `config_update` 命令远程更新。

## 算法依赖 (easyeai-api/)

通过 `api.cmake` 引入预编译库:

- `person_detect` - 人体检测 (RKNN)
- `face_detect` - 人脸检测 (RKNN)
- `face_detect_retian` - RetinaFace 模型
- `face_detect_scrfd` - SCRFD 模型
- `media/rga` - RGA 硬件加速图像处理
- `media/gst_opt` - GStreamer 媒体封装

## 板端硬件接口 (src/commonApi/)

- `camera.h` / `mipi_camera.c` - MIPI 相机驱动封装
- `gpio/` - GPIO 控制 (IR-CUT 切换)
- `uart/` - 串口通信
- `dma/` - DMA 内存分配
- `usb_camera/` - USB 相机支持 (含 VPU 解码)

## 当前研发重点

### 暗光与运动模糊优化

**问题** (见 `问题,解决方案,实际.md`):
- 暗光场景长曝光导致运动拖影
- 清晰度与模糊度评估在低光下不准确

**实验方案**:
- `src/utils/tasks/nafnet_tiny_enhancer.cpp` - NAFNet 去噪/去模糊
- `src/tools/convert_nafnet_tiny_to_rknn.py` - 模型转换工具
- 多帧融合增强 (已集成到 CameraTask)

**调参建议**:
- 优先调整 `device_config.json` 中的 `capture_defaults`
- 验证时使用 `camera_test_advanced` 查看 FPS 与清晰度统计
- 避免破坏主链路的上传与 TCP 控制功能

## 代码修改注意事项

1. **回调链路**: main.cpp 中的回调编排是核心,修改任务时需同步更新回调注册
2. **线程安全**: CameraTask/UploaderTask/TcpClient 各自独立线程,共享状态需加锁
3. **GPIO 操作**: IR-CUT 切换有最小间隔限制,避免频繁切换损坏硬件
4. **配置持久化**: 运行时参数变更需调用 `config.save()` 回写文件
5. **RKNN 模型路径**: 定义在 `src/include/main.h` 中的 `PERSON_MODEL_PATH` / `FACE_MODEL_PATH`

## 部署缺口

- `src/p_cam.service` 文件存在但为空,systemd 自启动尚未配置
- 板端日志轮转机制待完善
- 远程 OTA 更新流程待标准化
