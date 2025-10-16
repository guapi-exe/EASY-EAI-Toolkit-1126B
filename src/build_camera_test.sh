#!/bin/bash

echo "=========================================="
echo "编译摄像头帧测试程序"
echo "=========================================="

# 进入构建目录
cd "$(dirname "$0")"

# 创建或清理build目录
if [ ! -d "build" ]; then
    mkdir build
fi

cd build

# CMake配置
cmake ..

# 只编译camera_test
make camera_test -j$(nproc)

if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "✅ 编译成功!"
    echo "可执行文件: build/camera_test"
    echo "=========================================="
    echo "运行方式:"
    echo "  cd build && ./camera_test"
    echo "=========================================="
else
    echo "❌ 编译失败"
    exit 1
fi
