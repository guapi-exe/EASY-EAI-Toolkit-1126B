#!/bin/bash

set -e

# 编译参数
BUILD_DIR="build"
TARGET_NAME="test_rknn_init"

# 创建构建目录（如果不存在）
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

# CMake 配置（如果需要）
if [ ! -f "Makefile" ]; then
    cmake -DCMAKE_BUILD_TYPE=Release ..
fi

# 编译
make ${TARGET_NAME} -j$(nproc)

# 提示
echo ""
echo "==========================================="
echo "Build completed: ${BUILD_DIR}/${TARGET_NAME}"
echo "==========================================="
echo "To run on device:"
echo "  ./${TARGET_NAME}"
echo ""

