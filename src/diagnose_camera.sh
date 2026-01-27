#!/bin/bash
# CameraTask 诊断脚本

echo "=== CameraTask 诊断 ==="
echo ""

# 1. 检查模型文件
echo "1. 检查模型文件..."
if [ -f "person_detect.model" ]; then
    echo "  ✓ person_detect.model 存在"
else
    echo "  ✗ person_detect.model 不存在!"
fi

if [ -f "face_detect.model" ]; then
    echo "  ✓ face_detect.model 存在"
else
    echo "  ✗ face_detect.model 不存在!"
fi

# 2. 检查摄像头设备
echo ""
echo "2. 检查摄像头设备..."
if [ -c "/dev/video11" ]; then
    echo "  ✓ /dev/video11 存在"
else
    echo "  ✗ /dev/video11 不存在!"
fi

# 3. 检查是否有其他进程占用摄像头
echo ""
echo "3. 检查摄像头占用情况..."
CAMERA_PROCS=$(lsof /dev/video11 2>/dev/null)
if [ -n "$CAMERA_PROCS" ]; then
    echo "  ✗ 摄像头被占用:"
    echo "$CAMERA_PROCS"
else
    echo "  ✓ 摄像头未被占用"
fi

# 4. 检查库文件
echo ""
echo "4. 检查必需的库文件..."
libs=("librknnrt.so" "librga.so" "libopencv_core.so")
for lib in "${libs[@]}"; do
    if ldconfig -p | grep -q "$lib"; then
        echo "  ✓ $lib 可用"
    else
        echo "  ✗ $lib 不可用!"
    fi
done

# 5. 重新编译
echo ""
echo "5. 重新编译程序..."
cd /home/linaro/EASY-EAI-Toolkit-1126B/src
./build.sh
if [ $? -eq 0 ]; then
    echo "  ✓ 编译成功"
else
    echo "  ✗ 编译失败!"
    exit 1
fi

# 6. 运行诊断
echo ""
echo "6. 运行程序进行诊断..."
echo "   输出将包含详细的初始化日志"
echo "   按 Ctrl+C 可以停止"
echo ""

cd ../build
timeout 10 ./main 2>&1 | grep -E "CameraTask|ERROR|camera|Camera" | head -20

echo ""
echo "=== 诊断完成 ==="
echo ""
echo "常见问题解决方案:"
echo "1. 如果模型文件不存在，请检查路径或从正确位置复制"
echo "2. 如果摄像头设备不存在，检查硬件连接"
echo "3. 如果摄像头被占用，先停止其他程序："
echo "   pkill -9 camera_streaming"
echo "   pkill -9 main"
echo "4. 查看完整日志："
echo "   ./main 2>&1 | tee main.log"
