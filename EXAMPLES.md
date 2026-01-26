# 摄像头推流使用示例

本文档提供实际的命令行使用示例，帮助你快速上手。

## 📋 目录

- [基础推流](#基础推流)
- [高级推流](#高级推流)
- [性能优化](#性能优化)
- [多平台推流](#多平台推流)
- [实际应用场景](#实际应用场景)

## 基础推流

### 示例 1: 默认参数推流

最简单的使用方式，使用默认的 720p @ 25fps：

```bash
./camera_streaming --rtmp rtmp://localhost/live/stream
```

**输出日志示例**：
```
=== 摄像头推流程序 ===
摄像头分辨率: 3840x2160
输出分辨率: 1280x720 @ 25fps
推流方式: RTMP
推流地址: rtmp://localhost/live/stream
摄像头初始化成功
RTMP pipeline created: rtmp://localhost/live/stream
Streaming started
开始推流，按 Ctrl+C 停止...
=================================
推流统计: 总帧数=25, 当前FPS=25.00, 运行时间=1s
推流统计: 总帧数=50, 当前FPS=25.00, 运行时间=2s
...
```

### 示例 2: RTSP 推流

使用 RTSP 协议推流（适合监控场景）：

```bash
./camera_streaming --rtsp
```

### 示例 3: 推流到远程服务器

推流到远程流媒体服务器：

```bash
./camera_streaming --rtmp rtmp://192.168.1.100/live/mystream
```

### 示例 4: 保存本地视频

推流的同时保存本地视频文件：

```bash
./camera_streaming --rtmp rtmp://localhost/live/stream --save
```

视频将保存为 `output_<timestamp>.mp4`

## 高级推流

### 示例 5: 全高清推流

1080p @ 30fps 高清推流：

```bash
./camera_streaming \
    --rtmp rtmp://localhost/live/hd \
    --width 1920 \
    --height 1080 \
    --fps 30
```

**适用场景**：
- 高清监控
- 在线教育
- 直播活动

**网络要求**：至少 3 Mbps 上行带宽

### 示例 6: 2K 超清推流

2560x1440 @ 30fps 超清推流：

```bash
./camera_streaming \
    --rtmp rtmp://localhost/live/2k \
    --width 2560 \
    --height 1440 \
    --fps 30 \
    --save
```

**适用场景**：
- 专业视频制作
- 细节要求高的监控
- 大屏展示

**网络要求**：至少 5 Mbps 上行带宽

### 示例 7: 4K 推流

4K @ 25fps 推流（测试用）：

```bash
./camera_streaming \
    --rtmp rtmp://localhost/live/4k \
    --width 3840 \
    --height 2160 \
    --fps 25
```

**注意**：4K 推流需要强大的硬件和网络支持

## 性能优化

### 示例 8: 低带宽推流

适合网络条件较差的环境：

```bash
./camera_streaming \
    --rtmp rtmp://localhost/live/low \
    --width 854 \
    --height 480 \
    --fps 15
```

**特点**：
- 分辨率：480p
- 帧率：15fps
- 码率：约 500 kbps
- 网络要求：0.5 Mbps

### 示例 9: 低延迟推流

适合需要低延迟的场景：

```bash
./camera_streaming \
    --rtmp rtmp://localhost/live/lowlatency \
    --width 1280 \
    --height 720 \
    --fps 30
```

**配合播放**：
```bash
# 使用 HTTP-FLV 播放，延迟最低
vlc http://localhost:8080/live/lowlatency.flv
```

### 示例 10: 高帧率推流

适合运动场景或游戏直播：

```bash
./camera_streaming \
    --rtmp rtmp://localhost/live/highfps \
    --width 1920 \
    --height 1080 \
    --fps 60
```

**适用场景**：
- 体育赛事
- 游戏直播
- 快速运动物体追踪

## 多平台推流

### 示例 11: 推流到多个平台

使用脚本同时推流到多个平台：

```bash
#!/bin/bash
# multi_stream.sh

# 后台推流到平台 1
./camera_streaming --rtmp rtmp://platform1.com/live/stream1 &
PID1=$!

# 后台推流到平台 2
./camera_streaming --rtmp rtmp://platform2.com/live/stream2 &
PID2=$!

# 后台推流到平台 3
./camera_streaming --rtmp rtmp://platform3.com/live/stream3 &
PID3=$!

echo "多平台推流已启动"
echo "PID: $PID1, $PID2, $PID3"

# 等待退出信号
wait
```

### 示例 12: 推流到云平台

#### 推流到阿里云直播

```bash
./camera_streaming \
    --rtmp rtmp://push.aliyunlive.com/live/your_stream_key \
    --width 1920 \
    --height 1080 \
    --fps 30
```

#### 推流到腾讯云直播

```bash
./camera_streaming \
    --rtmp rtmp://push.tcloud.com/live/your_stream_key \
    --width 1920 \
    --height 1080 \
    --fps 30
```

#### 推流到 B 站直播

```bash
./camera_streaming \
    --rtmp rtmp://live-push.bilivideo.com/live-bvc/your_stream_key \
    --width 1920 \
    --height 1080 \
    --fps 30
```

**注意**：需要替换 `your_stream_key` 为实际的推流密钥

## 实际应用场景

### 场景 1: 视频监控系统

24小时连续推流，录制所有视频：

```bash
#!/bin/bash
# surveillance.sh

while true; do
    echo "$(date): 启动监控推流"
    
    ./camera_streaming \
        --rtmp rtmp://localhost/live/surveillance \
        --width 1920 \
        --height 1080 \
        --fps 25 \
        --save
    
    # 如果程序意外退出，等待 5 秒后重启
    echo "$(date): 推流程序退出，5秒后重启"
    sleep 5
done
```

**运行**：
```bash
chmod +x surveillance.sh
nohup ./surveillance.sh > surveillance.log 2>&1 &
```

### 场景 2: 在线教育直播

课堂实时直播：

```bash
# 上课时启动
./camera_streaming \
    --rtmp rtmp://edu-server.com/live/classroom_001 \
    --width 1920 \
    --height 1080 \
    --fps 30 \
    --save

# 保存的视频可用于课后回放
```

### 场景 3: 会议直播

会议室监控和直播：

```bash
#!/bin/bash
# meeting_stream.sh

MEETING_ID=$1
SAVE_DIR="/recordings/$(date +%Y%m%d)"

mkdir -p $SAVE_DIR
cd $SAVE_DIR

./camera_streaming \
    --rtmp rtmp://meeting-server.com/live/meeting_${MEETING_ID} \
    --width 1920 \
    --height 1080 \
    --fps 30 \
    --save

# 会议结束后，视频自动保存到指定目录
```

**使用**：
```bash
./meeting_stream.sh 20260126_001
```

### 场景 4: 智能门禁系统

人脸识别门禁推流：

```bash
./camera_streaming \
    --rtmp rtmp://localhost/live/entrance \
    --width 1280 \
    --height 720 \
    --fps 25
```

**配合使用**：
- 后端服务订阅视频流
- 实时进行人脸识别
- 记录进出记录

### 场景 5: 交通监控

道路交通实时监控：

```bash
./camera_streaming \
    --rtmp rtmp://traffic-center.com/live/camera_001 \
    --width 1920 \
    --height 1080 \
    --fps 25 \
    --save
```

### 场景 6: 工业检测

生产线质量检测：

```bash
./camera_streaming \
    --rtmp rtmp://factory-server.com/live/line_01 \
    --width 2560 \
    --height 1440 \
    --fps 30 \
    --save
```

**特点**：
- 高分辨率捕捉细节
- 保存视频用于事后分析
- 实时推流用于远程监控

## 系统集成

### 示例 13: 系统服务配置

创建 systemd 服务自动启动推流：

```bash
# /etc/systemd/system/camera-streaming.service

[Unit]
Description=Camera Streaming Service
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=root
WorkingDirectory=/root/camera-streaming
ExecStart=/root/camera-streaming/camera_streaming \
    --rtmp rtmp://localhost/live/stream \
    --width 1920 \
    --height 1080 \
    --fps 30
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

**启用服务**：
```bash
sudo systemctl daemon-reload
sudo systemctl enable camera-streaming
sudo systemctl start camera-streaming
sudo systemctl status camera-streaming
```

### 示例 14: Docker 容器化部署

创建 Dockerfile：

```dockerfile
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    libgstreamer1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    libopencv-dev

COPY camera_streaming /usr/local/bin/
COPY start.sh /usr/local/bin/

CMD ["/usr/local/bin/start.sh"]
```

**构建和运行**：
```bash
docker build -t camera-streaming .
docker run -d \
    --device /dev/video0 \
    --network host \
    --name camera-streaming \
    camera-streaming
```

## 监控和调试

### 示例 15: 带日志的推流

详细记录推流日志：

```bash
./camera_streaming \
    --rtmp rtmp://localhost/live/stream \
    --width 1920 \
    --height 1080 \
    --fps 30 \
    2>&1 | tee streaming_$(date +%Y%m%d_%H%M%S).log
```

### 示例 16: 性能监控

监控推流程序的性能：

```bash
#!/bin/bash
# monitor_streaming.sh

# 启动推流
./camera_streaming --rtmp rtmp://localhost/live/stream &
PID=$!

# 监控资源使用
while kill -0 $PID 2>/dev/null; do
    echo "$(date): CPU: $(ps -p $PID -o %cpu=)% MEM: $(ps -p $PID -o %mem=)%"
    sleep 5
done
```

## 故障恢复

### 示例 17: 自动重连推流

网络断开后自动重连：

```bash
#!/bin/bash
# auto_reconnect.sh

RTMP_URL="rtmp://server.com/live/stream"
MAX_RETRIES=999999

for i in $(seq 1 $MAX_RETRIES); do
    echo "$(date): 尝试推流 (第 $i 次)"
    
    ./camera_streaming --rtmp $RTMP_URL
    
    EXIT_CODE=$?
    echo "$(date): 推流退出 (退出码: $EXIT_CODE)"
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "正常退出"
        break
    else
        echo "异常退出，10秒后重试..."
        sleep 10
    fi
done
```

## 测试和验证

### 示例 18: 推流质量测试

测试不同参数下的推流质量：

```bash
#!/bin/bash
# quality_test.sh

RESOLUTIONS=("1280x720" "1920x1080" "2560x1440")
FRAMERATES=(15 25 30)

for res in "${RESOLUTIONS[@]}"; do
    for fps in "${FRAMERATES[@]}"; do
        WIDTH=$(echo $res | cut -d'x' -f1)
        HEIGHT=$(echo $res | cut -d'x' -f2)
        
        echo "测试: ${WIDTH}x${HEIGHT} @ ${fps}fps"
        
        timeout 60s ./camera_streaming \
            --rtmp rtmp://localhost/live/test \
            --width $WIDTH \
            --height $HEIGHT \
            --fps $fps \
            --save
        
        echo "---"
        sleep 5
    done
done
```

## 总结

本文档提供了 18 个实际使用示例，涵盖：

- ✅ 基础推流操作
- ✅ 高级配置选项
- ✅ 性能优化技巧
- ✅ 多平台推流
- ✅ 实际应用场景
- ✅ 系统集成
- ✅ 监控和调试
- ✅ 故障恢复

更多详细信息，请参考：
- [QUICKSTART.md](QUICKSTART.md) - 快速开始指南
- [README_STREAMING.md](README_STREAMING.md) - 完整文档

---

**提示**: 所有示例都可以根据实际需求进行调整和组合使用。
