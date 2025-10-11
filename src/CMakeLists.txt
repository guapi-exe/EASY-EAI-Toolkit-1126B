cmake_minimum_required(VERSION 3.10.2)

STRING(REGEX REPLACE ".*/(.*)" "\\1" CURRENT_FOLDER ${CMAKE_CURRENT_SOURCE_DIR})
MESSAGE("current project: " ${CURRENT_FOLDER})

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_CROSSCOMPILING TRUE)

cmake_host_system_information(RESULT arch_value QUERY OS_PLATFORM)
message(STATUS "architecture: " ${arch_value})

if((NOT "${arch_value}" STREQUAL "armv7l") AND (NOT "${arch_value}" STREQUAL "aarch64"))
    include ($ENV{HOME}/configs/cross.cmake)
endif()

find_package(CURL QUIET)

if (CURL_FOUND)
    message(STATUS "Found system libcurl: ${CURL_LIBRARIES}")
else()
    message(WARNING "CURL not found automatically, using manual include/lib paths")
    set(CURL_INCLUDE_DIRS ${CMAKE_SYSROOT}/usr/include)
    set(CURL_LIBRARIES ${CMAKE_SYSROOT}/usr/lib/libcurl.so)
endif()

include_directories(${CURL_INCLUDE_DIRS})
#若要使用${CMAKE_SYSROOT}变量，project必须要在set(CMAKE_TOOLCHAIN_FILE "${HOST_DIR}/share/buildroot/toolchainfile.cmake")后面设置
project(test_1)
add_definitions(-Wall -Wpointer-arith)

set(OpenCV_LIBS
    opencv_core 
    opencv_imgproc 
    opencv_imgcodecs
#    opencv_calib3d 
#    opencv_dnn 
#    opencv_features2d 
#    opencv_flann 
    opencv_highgui 
#    opencv_ml 
#    opencv_objdetect 
#    opencv_photo 
#    opencv_stitching
    opencv_videoio 
    opencv_video  
)

set(toolkit_root ${CMAKE_CURRENT_SOURCE_DIR}/../easyeai-api)
set(algorithm_root ${toolkit_root}/algorithm)
set(common_root    ${toolkit_root}/common)

include (${algorithm_root}/person_detect/api.cmake)
include (${algorithm_root}/face_detect/api.cmake)
include (${common_root}/base64/api.cmake)
include (${common_root}/json/api.cmake)

## api头文件路径
set(api_inc 
    include/
    include/tasks
    ${CJSON_INCLUDE_DIRS}
    ${BASE64_INCLUDE_DIRS} 
    ${FACE_DETECT_INCLUDE_DIRS} 
    ${PERSON_DETECT_INCLUDE_DIRS} 
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/logs
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/tinyekf
    ${CMAKE_CURRENT_SOURCE_DIR}/commonApi
    ${CMAKE_CURRENT_SOURCE_DIR}/commonApi/dma
    ${CMAKE_CURRENT_SOURCE_DIR}/commonApi/usb_camera
    )
## api源文件路径

file(GLOB api_srcs 
    ${BASE64_SOURCE_DIRS} 
    ${CJSON_SOURCE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/tasks/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/commonApi/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/commonApi/dma/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/commonApi/usb_camera/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/logs/*.c
)

## api静态库文件路径
link_directories(
   ${FACE_DETECT_LIBS_DIRS} 
   ${PERSON_DETECT_LIBS_DIRS} 
)
## 依赖的动态库文件
set(sysLib_list 
    ${BASE64_LIBS}
    ${CJSON_LIBS}
    ${PERSON_DETECT_LIBS} 
    ${FACE_DETECT_LIBS} 
    ${OpenCV_LIBS}
    rga 
    rockchip_mpp
    )

set(solutions ${CMAKE_CURRENT_SOURCE_DIR}/../../Solutions)
add_definitions(-DLOG_USE_COLOR)

#--------------------------
# Main executable
#--------------------------
set(project_name main)
add_executable(${project_name} main.cpp ${api_srcs} ${common_api_srcs})
target_link_libraries(${project_name} ${sysLib_list} ${CURL_LIBRARIES})
target_include_directories(${project_name} PRIVATE ${api_inc})
