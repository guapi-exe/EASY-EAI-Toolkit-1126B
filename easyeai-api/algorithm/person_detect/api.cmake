##在/usr/share/cmake-3.18/Modules/目录下的Find*.cmake，都能通过这种方式被找到
##具体的变量(如${OpenCV_INCLUDE_DIRS}、${OpenCV_LIBS})会被定义在里面的相对应的Find*.cmake文件中(通常就写在开头的描述里)
#find_package(OpenCV REQUIRED)
#
set(OpenCV_INCLUDE_DIRS
    ${CMAKE_SYSROOT}/usr/include/ 
    ${CMAKE_SYSROOT}/usr/include/opencv4/ 
)
set(OpenCV_LIBS_DIRS
    ${CAMKE_SYSROOT}/usr/lib/aarch64-linux-gnu/lapack
    ${CAMKE_SYSROOT}/usr/lib/aarch64-linux-gnu/blas
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

# person_detect 源文件（混合模式：使用.a库的解密 + 自编译的检测逻辑）
set(PERSON_DETECT_SOURCE_DIRS
    ${CMAKE_CURRENT_LIST_DIR}/person_detect.cpp
    ${CMAKE_CURRENT_LIST_DIR}/person_detect_postprocess.cpp
)

# static Library paths
set(PERSON_DETECT_LIBS_DIRS
    ${CMAKE_CURRENT_LIST_DIR}
    ${OpenCV_LIBS_DIRS}
    )

# headfile path
set(PERSON_DETECT_INCLUDE_DIRS
    ${OpenCV_INCLUDE_DIRS} 
    ${CMAKE_CURRENT_LIST_DIR} 
    )

# c/c++ flags（包含原始.a库以使用解密函数）
set(PERSON_DETECT_LIBS 
    person_detect
    rknnrt
    ${OpenCV_LIBS} 
    pthread
    stdc++ 
    )
