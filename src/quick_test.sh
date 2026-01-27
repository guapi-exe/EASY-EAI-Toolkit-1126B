#!/bin/bash
# 快速测试脚本

echo "=== CameraTask 快速测试 ==="
echo ""

cd /home/linaro/demo

# 1. 检查模型文件
echo "1. 检查模型文件..."
ls -lh person_detect.model face_detect.model 2>&1

# 2. 检查摄像头
echo ""
echo "2. 检查摄像头..."
ls -l /dev/video11

# 3. 检查是否有进程占用
echo ""
echo "3. 检查摄像头占用..."
lsof /dev/video11 2>&1 || echo "  未被占用"

# 4. 杀死可能的旧进程
echo ""
echo "4. 清理旧进程..."
pkill -9 main 2>/dev/null && echo "  已杀死旧的 main 进程" || echo "  无旧进程"
sleep 1

# 5. 运行程序（带超时）
echo ""
echo "5. 运行程序（30秒测试）..."
echo "   查看初始化日志..."
echo ""

timeout 30 stdbuf -oL -eL ./main 2>&1 | tee main_test.log

echo ""
echo "=== 测试完成 ==="
echo ""
echo "日志已保存到: main_test.log"
echo ""
echo "关键检查点:"
grep -E "CameraTask.*model|CameraTask.*camera|ERROR|WARN" main_test.log | head -20
