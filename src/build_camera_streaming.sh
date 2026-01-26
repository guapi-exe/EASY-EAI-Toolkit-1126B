#!/bin/bash
# 编译摄像头推流程序

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== 开始编译摄像头推流程序 ===${NC}"

# 进入 build 目录
cd "$(dirname "$0")"

# 检查 build 目录
if [ ! -d "../build" ]; then
    echo -e "${YELLOW}创建 build 目录...${NC}"
    mkdir -p ../build
fi

cd ../build

# 运行 CMake
echo -e "${YELLOW}运行 CMake...${NC}"
cmake ../src
if [ $? -ne 0 ]; then
    echo -e "${RED}CMake 配置失败!${NC}"
    exit 1
fi

# 编译
echo -e "${YELLOW}编译中...${NC}"
make camera_streaming -j$(nproc)
if [ $? -ne 0 ]; then
    echo -e "${RED}编译失败!${NC}"
    exit 1
fi

echo -e "${GREEN}=== 编译成功! ===${NC}"
echo -e "${GREEN}可执行文件: ../build/camera_streaming${NC}"
echo ""
echo -e "${YELLOW}使用示例:${NC}"
echo "  RTMP 推流: ./camera_streaming --rtmp rtmp://localhost/live/stream"
echo "  RTSP 推流: ./camera_streaming --rtsp"
echo "  自定义分辨率: ./camera_streaming --rtmp rtmp://localhost/live/stream --width 1920 --height 1080 --fps 30"
echo "  保存本地视频: ./camera_streaming --rtmp rtmp://localhost/live/stream --save"
