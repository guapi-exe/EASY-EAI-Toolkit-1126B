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

#若要使用${CMAKE_SYSROOT}变量，project必须要在set(CMAKE_TOOLCHAIN_FILE "${HOST_DIR}/share/buildroot/toolchainfile.cmake")后面设置
project(test_1)
add_definitions(-Wall -Wpointer-arith)

set(toolkit_root ${CMAKE_CURRENT_SOURCE_DIR}/../easyeai-api)
set(algorithm_root ${toolkit_root}/algorithm)
set(common_root    ${toolkit_root}/common)

include (${algorithm_root}/face_detect/api.cmake)
## api头文件路径
set(api_inc 
    include/
    ${FACE_DETECT_INCLUDE_DIRS} 
    ${CMAKE_CURRENT_SOURCE_DIR}/commonApi
    ${CMAKE_CURRENT_SOURCE_DIR}/commonApi/dma
    ${CMAKE_CURRENT_SOURCE_DIR}/commonApi/usb_camera
    )
## api源文件路径

file(GLOB api_srcs 
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp
)
## commonApi 源码
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/commonApi common_api_srcs)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/commonApi/dma common_api_srcs)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/commonApi/usb_camera common_api_srcs)
## api静态库文件路径
link_directories(
   ${FACE_DETECT_LIBS_DIRS} 
)
## 依赖的动态库文件
set(sysLib_list 
    ${FACE_DETECT_LIBS} 
    )

set(solutions ${CMAKE_CURRENT_SOURCE_DIR}/../../Solutions)

#--------------------------
# Main executable
#--------------------------
set(project_name main)
add_executable(${project_name} main.cpp ${api_srcs} ${common_api_srcs})
target_link_libraries(${project_name} ${sysLib_list})
target_include_directories(${project_name} PRIVATE ${api_inc})
