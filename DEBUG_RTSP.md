# RTSP 推流调试指南

## 问题：ffplay 连接但无法显示

### 当前状态
```
ffplay rtsp://192.168.1.83:8554/stream
...
nan : 0.000 fd=0 aq=0KB vq=0KB sq=0B 无法显示加载
```

这表明 ffplay 连接到了 RTSP 服务器，但没有收到视频数据。

## 调试步骤

### 1. 重新编译程序

最新代码已添加详细日志和修复：

```bash
cd /home/linaro/EASY-EAI-Toolkit-1126B/src
./build_camera_streaming.sh
cd ../build
```

### 2. 启动 RTSP 推流（注意观察日志）

```bash
./camera_streaming --rtsp --width 1920 --height 1080 --fps 30
```

你应该看到：
```
[INFO] === 摄像头推流程序 ===
[INFO] 摄像头初始化成功
[INFO] RTSP server created
[INFO] Stream ready at rtsp://localhost:8554/stream
[INFO] RTSP server started, waiting for connections...
[INFO] Use this command to view stream:
[INFO]   ffplay rtsp://localhost:8554/stream
[INFO]   or vlc rtsp://localhost:8554/stream
[INFO] 开始推流，按 Ctrl+C 停止...
```

### 3. 连接客户端并观察日志

**重要**: 当客户端连接时，服务器端应该输出：

```
[INFO] RTSP client connected, configuring media...
[INFO] Found appsrc, configuring...
[INFO] Client connected! Ready to stream at 1920x1080@30fps
[INFO] Starting to push frames to client...
```

**如果没有看到这些日志**，说明 `rtsp_media_configure` 回调没有被调用。

### 4. 客户端测试命令

#### 方法 1: ffplay with TCP transport（推荐）

```bash
ffplay -rtsp_transport tcp -fflags nobuffer -flags low_delay rtsp://192.168.1.83:8554/stream
```

参数说明：
- `-rtsp_transport tcp`: 使用 TCP 而不是 UDP，更稳定
- `-fflags nobuffer`: 不使用缓冲，降低延迟
- `-flags low_delay`: 低延迟模式

#### 方法 2: ffplay 详细模式

```bash
ffplay -v debug -rtsp_transport tcp rtsp://192.168.1.83:8554/stream
```

这会显示详细的调试信息，帮助诊断问题。

#### 方法 3: 使用 ffprobe 检查流

```bash
ffprobe -v error -show_format -show_streams rtsp://192.168.1.83:8554/stream
```

如果流正常，应该显示视频流信息：
```
[STREAM]
codec_name=h264
width=1920
height=1080
...
```

#### 方法 4: VLC

```bash
vlc --network-caching=300 --rtsp-tcp rtsp://192.168.1.83:8554/stream
```

#### 方法 5: GStreamer

```bash
gst-launch-1.0 -v rtspsrc location=rtsp://192.168.1.83:8554/stream protocols=tcp ! \
  rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink
```

### 5. 启用 GStreamer 调试

如果客户端连接了但服务器端没有日志，启用 GStreamer 调试：

```bash
# 设置 GStreamer 调试级别
export GST_DEBUG=3

# 或者更详细的 RTSP 调试
export GST_DEBUG=rtsp-server:5,rtsp-media:5,appsrc:5

# 然后运行程序
./camera_streaming --rtsp --width 1920 --height 1080 --fps 30
```

这会显示 GStreamer 内部的详细信息。

## 常见问题和解决方案

### 问题 1: 客户端连接但服务器无日志

**症状**: ffplay 显示连接但服务器端没有 "RTSP client connected" 日志

**可能原因**:
1. media-configure 信号没有正确连接
2. factory 配置有问题

**解决**: 重新编译最新代码，确保包含所有修复。

### 问题 2: 找不到 appsrc

**症状**: 日志显示 "Failed to find appsrc element 'videosrc'"

**解决**: 检查 launch_str 中的 appsrc 名称是否为 "videosrc"

### 问题 3: 推送失败

**症状**: "Failed to push buffer to appsrc"

**可能原因**:
1. caps 不匹配
2. 缓冲区满了
3. appsrc 没有正确配置

**解决**:
```bash
# 检查 appsrc 状态
export GST_DEBUG=appsrc:5
./camera_streaming --rtsp
```

### 问题 4: 网络连接问题

**检查防火墙**:
```bash
# 允许 8554 端口
iptables -I INPUT -p tcp --dport 8554 -j ACCEPT
iptables -I INPUT -p udp --dport 8554 -j ACCEPT

# 或者临时关闭防火墙测试
systemctl stop firewalld  # CentOS/RHEL
ufw disable  # Ubuntu
```

**检查端口是否监听**:
```bash
netstat -tuln | grep 8554
# 或
ss -tuln | grep 8554
```

## 测试步骤（完整流程）

### 在开发板上：

```bash
# 1. 编译
cd /home/linaro/EASY-EAI-Toolkit-1126B/src
./build_camera_streaming.sh

# 2. 启动（带 GStreamer 调试）
cd ../build
export GST_DEBUG=3
./camera_streaming --rtsp --width 1920 --height 1080 --fps 30
```

### 在 PC 上：

```bash
# 1. 首先测试连接
ping 192.168.1.83

# 2. 使用 ffprobe 检查流
ffprobe -v error -show_format -show_streams rtsp://192.168.1.83:8554/stream

# 3. 如果 ffprobe 成功，用 ffplay 播放
ffplay -rtsp_transport tcp rtsp://192.168.1.83:8554/stream

# 4. 如果还是有问题，启用详细日志
ffplay -v debug -rtsp_transport tcp rtsp://192.168.1.83:8554/stream 2>&1 | tee ffplay.log
```

## 预期的完整日志输出

### 服务器端（开发板）

```
[INFO] === 摄像头推流程序 ===
[INFO] 摄像头分辨率: 3840x2160
[INFO] 输出分辨率: 1920x1080 @ 30fps
[INFO] 推流方式: RTSP (UDP:5000)
[INFO] 摄像头初始化成功
[INFO] RTSP server created
[INFO] Stream ready at rtsp://localhost:8554/stream
[INFO] RTSP server started, waiting for connections...
[INFO] Use this command to view stream:
[INFO]   ffplay rtsp://localhost:8554/stream
[INFO]   or vlc rtsp://localhost:8554/stream
[INFO] 开始推流，按 Ctrl+C 停止...

# 当客户端连接时：
[INFO] RTSP client connected, configuring media...
[INFO] Found appsrc, configuring...
[INFO] Client connected! Ready to stream at 1920x1080@30fps
[INFO] Starting to push frames to client...

# 定期 FPS 统计：
[INFO] FPS: 30.00
[INFO] FPS: 30.00
...
```

### 客户端（PC）

```
ffplay version ... Copyright (c) 2003-2026 the FFmpeg developers
Input #0, rtsp, from 'rtsp://192.168.1.83:8554/stream':
  Metadata:
    title           : Session streamed with GStreamer
  Duration: N/A, start: 0.000000, bitrate: N/A
    Stream #0:0: Video: h264 (High), yuv420p, 1920x1080, 30 fps, 30 tbr, 90k tbn
# 然后应该显示视频窗口
```

## 如果还是不行

### 1. 简化测试

尝试降低分辨率和帧率：
```bash
./camera_streaming --rtsp --width 640 --height 480 --fps 15
```

### 2. 使用本地测试

在开发板上同时运行服务器和客户端：
```bash
# Terminal 1
./camera_streaming --rtsp --width 640 --height 480 --fps 15

# Terminal 2
ffplay rtsp://localhost:8554/stream
```

### 3. 检查 OpenCV 和 GStreamer 版本

```bash
# 检查 GStreamer
gst-inspect-1.0 --version

# 检查 rtsp-server 插件
gst-inspect-1.0 rtsp-server

# 检查 appsrc
gst-inspect-1.0 appsrc
```

### 4. 测试简单的 RTSP 推流

使用 GStreamer 命令行测试 RTSP server 是否正常：

```bash
# 启动一个测试 RTSP 服务器
gst-rtsp-server-1.0 \
  --mount-point=/test \
  --gst-launch="videotestsrc ! x264enc ! rtph264pay name=pay0"

# 在另一个终端测试
ffplay rtsp://localhost:8554/test
```

如果这个能工作，说明 RTSP server 本身没问题，问题在于摄像头数据流。

## 下一步

请重新编译并运行，然后提供：
1. **服务器端完整日志**（从启动到客户端连接的所有输出）
2. **客户端详细日志**（使用 `ffplay -v debug` 的输出）
3. **ffprobe 输出**（如果能连接的话）

这些信息可以帮助准确定位问题。
