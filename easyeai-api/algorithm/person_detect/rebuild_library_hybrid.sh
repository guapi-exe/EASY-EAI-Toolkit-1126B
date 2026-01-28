#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="$SCRIPT_DIR"
LIB_FILE="$LIB_DIR/libperson_detect.a"
BACKUP_FILE="$LIB_DIR/libperson_detect.a.old"

echo "=========================================="
echo "重构 person_detect 静态库（保留解密函数）"
echo "=========================================="

# 检查备份是否存在
if [ ! -f "$BACKUP_FILE" ]; then
    echo "错误: 未找到原始库备份 $BACKUP_FILE"
    exit 1
fi

# 1. 创建临时目录
TEMP_DIR=$(mktemp -d)
echo "[1/6] 创建临时目录: $TEMP_DIR"

# 2. 提取原始库的所有 .o 文件
echo "[2/6] 从原库提取 .o 文件..."
cd "$TEMP_DIR"
ar x "$BACKUP_FILE"
OBJ_COUNT=$(ls -1 *.o 2>/dev/null | wc -l)
echo "  ✓ 提取了 $OBJ_COUNT 个目标文件"

# 3. 编译新的 person_detect_postprocess.cpp
echo "[3/6] 编译 person_detect_postprocess.cpp..."
cd "$LIB_DIR"

OPENCV_CFLAGS=$(pkg-config --cflags opencv4 2>/dev/null || echo "-I/usr/include/opencv4")

g++ -c -fPIC -O2 -std=c++11 $OPENCV_CFLAGS \
    -I. \
    person_detect_postprocess.cpp \
    -o "$TEMP_DIR/person_detect_postprocess.cpp.o"

if [ $? -eq 0 ]; then
    echo "  ✓ person_detect_postprocess.cpp.o 编译成功（替换原文件）"
else
    echo "  ✗ person_detect_postprocess.cpp.o 编译失败!"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# 4. 编译新的 person_detect.cpp
echo "[4/6] 编译 person_detect.cpp..."

g++ -c -fPIC -O2 -std=c++11 $OPENCV_CFLAGS \
    -I. \
    person_detect.cpp \
    -o "$TEMP_DIR/person_detect.cpp.o"

if [ $? -eq 0 ]; then
    echo "  ✓ person_detect.cpp.o 编译成功（替换原文件）"
else
    echo "  ✗ person_detect.cpp.o 编译失败!"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# 5. 重新打包静态库
echo "[5/6] 重新打包静态库..."
cd "$TEMP_DIR"
ar rcs "$LIB_FILE" *.o

if [ $? -eq 0 ]; then
    NEW_OBJ_COUNT=$(ls -1 *.o 2>/dev/null | wc -l)
    echo "  ✓ 库重建成功，包含 $NEW_OBJ_COUNT 个目标文件"
    echo "    - decryption.c.o (来自原始库，保留硬件验证和解密逻辑)"
    echo "    - person_detect_postprocess.cpp.o (新编译，纯C++11算法)"
    echo "    - person_detect.cpp.o (新编译，OpenCV 4.6兼容)"
else
    echo "  ✗ 库重建失败!"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# 6. 清理
echo "[6/6] 清理临时文件..."
rm -rf "$TEMP_DIR"

echo ""
echo "=========================================="
echo "✓ 库重构完成!"
echo "=========================================="
echo "原库备份: $BACKUP_FILE"
echo "新库位置: $LIB_FILE"
echo ""
echo "说明: 保留原始解密函数，只替换检测逻辑"
echo ""
