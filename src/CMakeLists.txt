cmake_minimum_required(VERSION 3.10.2)

STRING(REGEX REPLACE ".*/(.*)" "\\1" CURRENT_FOLDER ${CMAKE_CURRENT_SOURCE_DIR})
MESSAGE("current project: " ${CURRENT_FOLDER})

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

# Find OpenCV package
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(toolkit_root ${CMAKE_CURRENT_SOURCE_DIR}/../easyeai-api)
set(algorithm_root ${toolkit_root}/algorithm)
set(common_root    ${toolkit_root}/common)
set(media_root     ${toolkit_root}/media)

include (${algorithm_root}/person_detect/api.cmake)
include (${algorithm_root}/face_detect/api.cmake)
include (${algorithm_root}/face_detect_retian/api.cmake)
include (${algorithm_root}/face_detect_scrfd/api.cmake)
include (${common_root}/base64/api.cmake)
include (${common_root}/json/api.cmake)
include (${media_root}/rga/api.cmake)
include (${media_root}/gst_opt/api.cmake)

## api头文件路径
set(api_inc 
    include/
    include/tasks
    ${CJSON_INCLUDE_DIRS}
    ${BASE64_INCLUDE_DIRS} 
    ${FACE_DETECT_INCLUDE_DIRS} 
    ${FACE_DETECT_RETIAN_INCLUDE_DIRS}
    ${FACE_DETECT_SCRFD_INCLUDE_DIRS}
    ${PERSON_DETECT_INCLUDE_DIRS}
    ${RGA_INCLUDE_DIRS}
    ${GSTOPT_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/logs
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/json
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/tinyekf
    ${CMAKE_CURRENT_SOURCE_DIR}/commonApi
    ${CMAKE_CURRENT_SOURCE_DIR}/commonApi/dma
    ${CMAKE_CURRENT_SOURCE_DIR}/commonApi/uart
    ${CMAKE_CURRENT_SOURCE_DIR}/commonApi/gpio
    ${CMAKE_CURRENT_SOURCE_DIR}/commonApi/usb_camera
    )
## api源文件路径

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

# RetinaFace 源文件
set(face_detect_retian_srcs
    ${algorithm_root}/face_detect_retian/face_detect_retian.cpp
    ${algorithm_root}/face_detect_scrfd/face_detect_scrfd.cpp
)

# SCRFD 源文件
set(face_detect_scrfd_srcs
)

## api静态库文件路径
link_directories(
   ${FACE_DETECT_LIBS_DIRS} 
   ${FACE_DETECT_RETIAN_LIBS_DIRS}
   ${FACE_DETECT_SCRFD_LIBS_DIRS}
   ${PERSON_DETECT_LIBS_DIRS}
   ${GSTOPT_LIBS_DIRS}
)
## 依赖的动态库文件
set(sysLib_list 
    ${BASE64_LIBS}
    ${CJSON_LIBS}
    ${RGA_LIBS}
    ${PERSON_DETECT_LIBS} 
    ${FACE_DETECT_LIBS}
    ${GSTOPT_LIBS}
    ${OpenCV_LIBRARIES}
    rga 
    gpiod
    rockchip_mpp
    )

## RetinaFace 专用库列表（用于 camera_test_retian）
set(retian_lib_list
    ${sysLib_list}
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

#--------------------------
# Camera Test executable
#--------------------------
#set(camera_test_name camera_test)
#add_executable(${camera_test_name} camera_test.cpp ${face_detect_retian_srcs} ${api_srcs} ${common_api_srcs})
#target_link_libraries(${camera_test_name} ${retian_lib_list} ${CURL_LIBRARIES})
#target_include_directories(${camera_test_name} PRIVATE ${api_inc})

#--------------------------
# Camera Test Advanced executable
#--------------------------
#set(camera_test_advanced_name camera_test_advanced)
#add_executable(${camera_test_advanced_name} camera_test_advanced.cpp ${face_detect_retian_srcs} ${api_srcs} ${common_api_srcs})
#target_link_libraries(${camera_test_advanced_name} ${retian_lib_list} ${CURL_LIBRARIES})
#target_include_directories(${camera_test_advanced_name} PRIVATE ${api_inc})

#--------------------------
# Camera Test RetinaFace executable
#--------------------------
#set(camera_test_retian_name camera_test_retian)
#add_executable(${camera_test_retian_name} camera_test_retian.cpp ${face_detect_retian_srcs} ${api_srcs} ${common_api_srcs})
#target_link_libraries(${camera_test_retian_name} ${retian_lib_list} ${CURL_LIBRARIES})
#target_include_directories(${camera_test_retian_name} PRIVATE ${api_inc})

#--------------------------
# Camera Test SCRFD executable
#--------------------------
#set(camera_test_scrfd_name camera_test_scrfd)
#add_executable(${camera_test_scrfd_name} camera_test_scrfd.cpp ${face_detect_scrfd_srcs} ${face_detect_retian_srcs} ${api_srcs} ${common_api_srcs})
#target_link_libraries(${camera_test_scrfd_name} ${retian_lib_list} ${CURL_LIBRARIES})
#target_include_directories(${camera_test_scrfd_name} PRIVATE ${api_inc})

#--------------------------
# Camera Streaming executable
#--------------------------
#set(camera_streaming_name camera_streaming)
#add_executable(${camera_streaming_name} camera_streaming.cpp ${api_srcs} ${common_api_srcs})
#target_link_libraries(${camera_streaming_name} ${sysLib_list} ${CURL_LIBRARIES})
#target_include_directories(${camera_streaming_name} PRIVATE ${api_inc})

#--------------------------
# RKNN Init Test executable
#--------------------------
set(test_rknn_name test_rknn_init)
add_executable(${test_rknn_name} 
    test_rknn_init.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/logs/log.c
)
target_link_libraries(${test_rknn_name} ${PERSON_DETECT_LIBS} ${FACE_DETECT_LIBS})
target_include_directories(${test_rknn_name} PRIVATE 
    ${PERSON_DETECT_INCLUDE_DIRS}
    ${FACE_DETECT_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/logs
)