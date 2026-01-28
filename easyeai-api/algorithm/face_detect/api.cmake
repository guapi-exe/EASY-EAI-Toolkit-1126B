##在/usr/share/cmake-3.18/Modules/目录下的Find*.cmake，都能通过这种方式被找到
##具体的变量(如${OpenCV_INCLUDE_DIRS}、${OpenCV_LIBS})会被定义在里面的相对应的Find*.cmake文件中(通常就写在开头的描述里)
#find_package(OpenCV REQUIRED)
#
set(OpenCV_INCLUDE_DIRS
    ${CMAKE_SYSROOT}/usr/include/ 
    ${CMAKE_SYSROOT}/usr/include/opencv4/ 
)
set(OpenCV_LIBS_DIRS
    ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/lapack
    ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/blas
)
set(OpenCV_LIBS
    opencv_core 
    opencv_imgproc 
    opencv_imgcodecs
#    opencv_calib3d 
#    opencv_dnn 
#    opencv_features2d 
#    opencv_flann 
#    opencv_highgui 
#    opencv_ml 
#    opencv_objdetect 
#    opencv_photo 
#    opencv_stitching
#    opencv_videoio 
#    opencv_video  
)

# face_detect 源文件（使用.a库的解密 + 自编译的完整检测逻辑）
set(FACE_DETECT_SOURCE_DIRS
    ${CMAKE_CURRENT_LIST_DIR}/hardward_verify_override.c
    ${CMAKE_CURRENT_LIST_DIR}/generator.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tools.cpp
    ${CMAKE_CURRENT_LIST_DIR}/decode.cpp
    ${CMAKE_CURRENT_LIST_DIR}/face_detect.cpp
)

# static Library paths
set(FACE_DETECT_LIBS_DIRS
    ${CMAKE_CURRENT_LIST_DIR}
    ${OpenCV_LIBS_DIRS}
    )

# headfile path
set(FACE_DETECT_INCLUDE_DIRS
    ${OpenCV_INCLUDE_DIRS} 
    ${CMAKE_CURRENT_LIST_DIR} 
    )

# c/c++ flags（包含原始.a库仅用于解密功能：decrypte_init和decrypte_model）
set(FACE_DETECT_LIBS 
    face_detect
    rknnrt
    ${OpenCV_LIBS} 
    pthread
    stdc++ 
    )
