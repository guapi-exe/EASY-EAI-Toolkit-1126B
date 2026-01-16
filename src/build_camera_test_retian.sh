#!/bin/bash

# RetinaFace 人脸检测测试程序编译脚本

echo "========================================"
echo "编译 camera_test_retian"
echo "========================================"

# 进入 build 目录
cd ../build

# 运行 CMake 配置（如果需要）
if [ ! -f "Makefile" ]; then
    echo "运行 CMake 配置..."
    cmake ../src
fi

# 编译 camera_test_retian
echo "开始编译..."
make camera_test_retian -j$(nproc)

if [ $? -eq 0 ]; then
    echo "========================================"
    echo "编译成功！"
    echo "可执行文件: ../build/camera_test_retian"
    echo "========================================"
    echo ""
    echo "使用说明:"
    echo "  ./camera_test_retian --help"
    echo ""
    echo "示例:"
    echo "  ./camera_test_retian --model ./retinaface_480x640.rknn --save"
    echo "  ./camera_test_retian --model ./rfb_480x640.rknn --type 2 --conf 0.6"
    echo ""
else
    echo "========================================"
    echo "编译失败！"
    echo "========================================"
    exit 1
fi
