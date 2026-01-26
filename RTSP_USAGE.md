# RTSP 推流使用说明

## 完整 RTSP 推流实现说明

本程序已实现基于 GstRTSPServer 的完整 RTSP 推流功能。

### 工作原理

1. **RTSP Server 模式**: 程序创建一个 RTSP Server，监听端口 8554
2. **延迟连接**: appsrc 只在客户端连接时才创建（通过 media_configure 回调）
3. **自动适配**: 在没有客户端连接时，程序会持续运行但不推送数据
4. **客户端连接**: 当客户端连接时，自动创建 pipeline 并开始推流

## 编译程序

```bash
cd src
./build_camera_streaming.sh
```

编译成功后，可执行文件位于 `build/camera_streaming`

## 运行 RTSP 推流

### 基本命令

```bash
# 使用默认参数（1920x1080@30fps）
./camera_streaming --rtsp

# 指定分辨率和帧率
./camera_streaming --rtsp --width 1920 --height 1080 --fps 30

# 同时保存到本地文件
./camera_streaming --rtsp --width 1920 --height 1080 --fps 30 --save
```

### 参数说明

- `--rtsp`: 启用 RTSP 推流模式
- `--width`: 输出视频宽度（默认 1920）
- `--height`: 输出视频高度（默认 1080）
- `--fps`: 输出视频帧率（默认 30）
- `--save`: 同时保存视频到本地 MP4 文件

## 查看 RTSP 流

### 方法 1: 使用 ffplay

```bash
# 在同一设备或其他设备上运行
ffplay rtsp://设备IP地址:8554/stream

# 如果在同一设备上测试
ffplay rtsp://localhost:8554/stream
```

### 方法 2: 使用 VLC

```bash
vlc rtsp://设备IP地址:8554/stream
```

或者在 VLC 中：
1. 打开 VLC
2. 媒体 → 打开网络串流
3. 输入: `rtsp://设备IP地址:8554/stream`
4. 点击播放

### 方法 3: 使用 GStreamer

```bash
gst-launch-1.0 rtspsrc location=rtsp://设备IP地址:8554/stream ! rtph264depay ! h264parse ! avdec_h264 ! autovideosink
```

### 方法 4: 使用 OpenCV (Python)

```python
import cv2

# 连接 RTSP 流
cap = cv2.VideoCapture('rtsp://设备IP地址:8554/stream')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('RTSP Stream', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 运行流程

### 1. 启动程序

```bash
root@linaro-alip:/home/linaro/demo# ./camera_streaming --rtsp --width 1920 --height 1080 --fps 30
```

你会看到类似输出：
```
09:03:34 [INFO] === 摄像头推流程序 ===
09:03:34 [INFO] 摄像头分辨率: 3840x2160
09:03:34 [INFO] 输出分辨率: 1920x1080 @ 30fps
09:03:34 [INFO] 推流方式: RTSP (UDP:5000)
09:03:34 [INFO] 摄像头初始化成功
09:03:34 [INFO] RTSP server created
09:03:34 [INFO] Stream ready at rtsp://localhost:8554/stream
09:03:34 [INFO] RTSP server started, waiting for connections...
09:03:34 [INFO] Use this command to view stream:
09:03:34 [INFO]   ffplay rtsp://localhost:8554/stream
09:03:34 [INFO]   or vlc rtsp://localhost:8554/stream
09:03:34 [INFO] 开始推流，按 Ctrl+C 停止...
```

### 2. 连接播放器

在另一个终端或其他设备上运行播放器：
```bash
ffplay rtsp://192.168.1.100:8554/stream
```

**重要**: 
- 在客户端连接之前，程序会持续运行但不消耗太多资源
- 当客户端连接时，会自动创建 media pipeline 并开始推流
- 可以同时有多个客户端连接观看（shared media）

### 3. 停止推流

按 `Ctrl+C` 停止程序：
```
09:05:34 [INFO] Streaming stopped
09:05:34 [INFO] =================================
09:05:34 [INFO] 推流结束统计:
09:05:34 [INFO]   总帧数: 3600
09:05:34 [INFO]   运行时间: 120 秒
09:05:34 [INFO]   平均FPS: 30.00
09:05:34 [INFO] =================================
```

## 常见问题

### 问题 1: 看到 "推送帧失败" 错误

**原因**: 这个错误已经被修复。新版本在没有客户端连接时不会报错。

**解决**: 重新编译并运行最新版本。

### 问题 2: 客户端无法连接

**检查**:
1. 确认防火墙允许 8554 端口
   ```bash
   # 如果使用 iptables
   iptables -I INPUT -p tcp --dport 8554 -j ACCEPT
   ```

2. 确认设备 IP 地址
   ```bash
   ifconfig
   # 或
   ip addr show
   ```

3. 使用正确的 IP 地址（不是 localhost）

### 问题 3: 视频卡顿或延迟高

**优化**:
1. 降低分辨率或帧率
   ```bash
   ./camera_streaming --rtsp --width 1280 --height 720 --fps 25
   ```

2. 确保网络稳定（如果远程观看）

3. 检查编码器设置（当前使用 x264enc zerolatency 模式）

### 问题 4: 多客户端连接

当前实现支持多个客户端同时连接（shared media），所有客户端会看到相同的流。

## 网络访问

### 局域网访问

如果要在局域网内其他设备上观看：

1. 查找设备 IP 地址：
   ```bash
   ifconfig eth0  # 或 wlan0
   ```

2. 在其他设备上使用该 IP：
   ```bash
   ffplay rtsp://192.168.1.100:8554/stream
   ```

### 公网访问

如果需要从公网访问（不推荐直接暴露）：

1. 配置路由器端口转发: 外部端口 8554 → 设备 IP:8554
2. 使用动态 DNS 服务获取域名
3. 使用域名和端口访问
4. **建议**: 使用 VPN 或隧道服务（如 frp, ngrok）更安全

## 性能参数

- **延迟**: 约 100-300ms（取决于网络和编码设置）
- **码率**: 2000 kbps（在代码中可调整）
- **编码器**: x264enc with zerolatency tune
- **传输协议**: RTP over UDP（默认）

## 技术细节

### Pipeline 结构

```
appsrc (videosrc) 
  → videoconvert 
  → video/x-raw,format=I420 
  → x264enc (zerolatency, 2000kbps) 
  → rtph264pay 
  → RTSP Server
```

### 关键实现

1. **GstRTSPServer**: RTSP 服务器管理
2. **GstRTSPMediaFactory**: Media 工厂模式
3. **media-configure 回调**: 客户端连接时配置 appsrc
4. **appsrc**: 接收摄像头帧数据
5. **x264enc**: H.264 硬件/软件编码
6. **rtph264pay**: H.264 RTP 打包

### 代码流程

```
1. create_rtsp_pipeline() 
   → 创建 RTSP Server
   → 创建 Media Factory
   → 设置 launch string
   → 连接 media-configure 信号
   
2. rtsp_media_configure() [客户端连接时调用]
   → 从 media 获取 appsrc
   → 配置 caps (BGR, 分辨率, 帧率)
   → 保存 appsrc 引用到 StreamContext
   
3. push_frame()
   → 检查是否有客户端连接 (appsrc != NULL)
   → 调整帧大小
   → 创建 GstBuffer
   → 推送到 appsrc
```

## 对比 RTMP 推流

| 特性 | RTSP | RTMP |
|------|------|------|
| 延迟 | 低 (100-300ms) | 中 (1-3s) |
| 服务器 | 内置 | 需要外部服务器 |
| 协议 | RTP/RTCP | TCP |
| 播放器 | VLC, ffplay, 原生支持 | 需要 Flash 或特殊播放器 |
| 适用场景 | 实时监控、对讲 | 直播、录播 |

## 总结

新的 RTSP 实现：
- ✅ 完整的 RTSP Server 功能
- ✅ 支持多客户端连接
- ✅ 低延迟实时传输
- ✅ 客户端连接时自动创建 pipeline
- ✅ 无客户端时不报错，持续运行
- ✅ 标准 RTSP 协议，兼容各种播放器

## 下一步

可以进一步优化：
1. 添加硬件编码支持 (mpph264enc)
2. 支持音频流
3. 添加认证机制
4. 支持 RTSP over TCP
5. 添加录制功能
6. 实现双向对讲
