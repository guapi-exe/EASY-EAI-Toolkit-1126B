#!/bin/bash
# 快速测试内存泄漏修复

echo "=== 内存泄漏修复测试 ==="
echo ""

# 重新编译
echo "1. 重新编译程序..."
cd /home/linaro/EASY-EAI-Toolkit-1126B/src
./build_camera_streaming.sh
if [ $? -ne 0 ]; then
    echo "编译失败!"
    exit 1
fi

cd ../build

echo ""
echo "2. 启动程序并监控内存..."
echo "   预期：内存应在 50-100MB 稳定，不再每秒增长 6-12MB"
echo ""

# 启动程序
./camera_streaming --rtsp --width 1920 --height 1080 --fps 30 &
PID=$!

echo "程序 PID: $PID"
echo ""
echo "监控 2 分钟..."
echo "时间(s) | 内存(MB) | 增长(MB)"
echo "--------|---------|----------"

LAST_MEM=0
START_TIME=$(date +%s)

for i in {1..120}; do
    sleep 1
    
    if ! ps -p $PID > /dev/null 2>&1; then
        echo ""
        echo "程序已退出!"
        break
    fi
    
    MEM=$(ps -p $PID -o rss --no-headers)
    MEM_MB=$(echo "scale=2; $MEM / 1024" | bc)
    
    if [ $LAST_MEM -ne 0 ]; then
        DIFF=$(echo "scale=2; $MEM_MB - $LAST_MEM" | bc)
        echo "$i | $MEM_MB | $DIFF"
    else
        echo "$i | $MEM_MB | -"
    fi
    
    LAST_MEM=$MEM_MB
    
    # 每 30 秒显示一次统计
    if [ $((i % 30)) -eq 0 ]; then
        echo ""
        echo "--- $i 秒检查点 ---"
        if [ $(echo "$MEM_MB > 200" | bc) -eq 1 ]; then
            echo "警告: 内存超过 200MB!"
        elif [ $(echo "$MEM_MB < 150" | bc) -eq 1 ]; then
            echo "良好: 内存控制在 150MB 以下"
        fi
        echo ""
    fi
done

echo ""
echo "3. 停止程序..."
kill -SIGINT $PID 2>/dev/null
sleep 2
kill -9 $PID 2>/dev/null

echo ""
echo "测试完成!"
echo ""
echo "如果内存稳定在 50-100MB，说明泄漏已修复。"
echo "如果内存仍然持续增长，请查看程序日志中的详细信息。"
