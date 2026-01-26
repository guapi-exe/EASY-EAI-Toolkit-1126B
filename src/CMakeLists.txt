cmake_minimum_required(VERSION 3.10.2)

project(test_1 LANGUAGES C CXX)

# ==========================================
# 架构判断
# ==========================================
cmake_host_system_information(RESULT arch_value QUERY OS_PLATFORM)
message(STATUS "architecture: ${arch_value}")

# 只有在非 ARM 本机时才启用交叉工具链
if(NOT "${arch_value}" STREQUAL "armv7l" AND
   NOT "${arch_value}" STREQUAL "aarch64")
    include($ENV{HOME}/configs/cross.cmake)
endif()

# ==========================================
# 编译器 & 稳定性设置（防 gcc ICE）
# ==========================================
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ARM + gcc-11/12 强烈建议这样
add_compile_options(
    -O0
    -g
    -fno-strict-aliasing
)

add_definitions(
    -Wall
    -Wpointer-arith
    -DLOG_USE_COLOR
)

# ==========================================
# CURL
# ==========================================
find_package(CURL QUIET)
if (CURL_FOUND)
    message(STATUS "Found system libcurl: ${CURL_LIBRARIES}")
else()
    message(WARNING "CURL not found automatically, using manual paths")
    set(CURL_INCLUDE_DIRS ${CMAKE_SYSROOT}/usr/include)
    set(CURL_LIBRARIES ${CMAKE_SYSROOT}/usr/lib/libcurl.so)
endif()

# ==========================================
# OpenCV libs
# ==========================================
set(OpenCV_LIBS
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_highgui
    opencv_videoio
    opencv_video
)

# ==========================================
# Toolkit paths
# ==========================================
set(toolkit_root ${CMAKE_CURRENT_SOURCE_DIR}/../easyeai-api)
set(algorithm_root ${toolkit_root}/algorithm)
set(common_root    ${toolkit_root}/common)
set(media_root     ${toolkit_root}/media)

# ==========================================
# SDK api.cmake
# ==========================================
include(${algorithm_root}/person_detect/api.cmake)
include(${algorithm_root}/face_detect/api.cmake)
include(${algorithm_root}/face_detect_retian/api.cmake)
include(${algorithm_root}/face_detect_scrfd/api.cmake)
include(${common_root}/base64/api.cmake)
include(${common_root}/json/api.cmake)
include(${media_root}/rga/api.cmake)

# ==========================================
# Include dirs
# ==========================================
set(api_inc
    include
    include/tasks
    ${CJSON_INCLUDE_DIRS}
    ${BASE64_INCLUDE_DIRS}
    ${FACE_DETECT_INCLUDE_DIRS}
    ${FACE_DETECT_RETIAN_INCLUDE_DIRS}
    ${FACE_DETECT_SCRFD_INCLUDE_DIRS}
    ${PERSON_DETECT_INCLUDE_DIRS}
    ${RGA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/logs
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/json
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/tinyekf
    ${CMAKE_CURRENT_SOURCE_DIR}/commonApi
    ${CMAKE_CURRENT_SOURCE_DIR}/commonApi/dma
    ${CMAKE_CURRENT_SOURCE_DIR}/commonApi/uart
    ${CMAKE_CURRENT_SOURCE_DIR}/commonApi/gpio
    ${CMAKE_CURRENT_SOURCE_DIR}/commonApi/usb_camera
)

# ==========================================
# Sources
# ==========================================
file(GLOB api_srcs
    ${BASE64_SOURCE_DIRS}
    ${CJSON_SOURCE_DIRS}
    ${RGA_SOURCE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/tasks/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/commonApi/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/commonApi/dma/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/commonApi/uart/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/commonApi/gpio/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/commonApi/usb_camera/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/logs/*.c
)

set(face_detect_retian_srcs
    ${algorithm_root}/face_detect_retian/face_detect_retian.cpp
    ${algorithm_root}/face_detect_scrfd/face_detect_scrfd.cpp
)

# ==========================================
# Link directories
# ==========================================
link_directories(
    ${FACE_DETECT_LIBS_DIRS}
    ${FACE_DETECT_RETIAN_LIBS_DIRS}
    ${FACE_DETECT_SCRFD_LIBS_DIRS}
    ${PERSON_DETECT_LIBS_DIRS}
)

# ==========================================
# System libs
# ==========================================
set(sysLib_list
    ${BASE64_LIBS}
    ${CJSON_LIBS}
    ${RGA_LIBS}
    ${PERSON_DETECT_LIBS}
    ${FACE_DETECT_LIBS}
    ${OpenCV_LIBS}
    rga
    gpiod
    rockchip_mpp
)

# ==========================================
# SDK 静态库（关键：避免 gcc ICE）
# ==========================================
add_library(easyeai_api STATIC
    ${api_srcs}
)

target_include_directories(easyeai_api PUBLIC ${api_inc})
target_link_libraries(easyeai_api PUBLIC ${sysLib_list})

# ==========================================
# Main executable
# ==========================================
add_executable(main main.cpp)
target_link_libraries(main easyeai_api ${CURL_LIBRARIES})

# ==========================================
# Camera test advanced
# ==========================================
add_executable(camera_test_advanced
    camera_test_advanced.cpp
    ${face_detect_retian_srcs}
)

target_link_libraries(camera_test_advanced
    easyeai_api
    ${CURL_LIBRARIES}
)
