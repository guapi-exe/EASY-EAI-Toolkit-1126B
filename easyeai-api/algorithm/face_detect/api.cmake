##��/usr/share/cmake-3.18/Modules/Ŀ¼�µ�Find*.cmake������ͨ�����ַ�ʽ���ҵ�
##����ı���(��${OpenCV_INCLUDE_DIRS}��${OpenCV_LIBS})�ᱻ��������������Ӧ��Find*.cmake�ļ���(ͨ����д�ڿ�ͷ��������)
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
    opencv_highgui 
#    opencv_ml 
#    opencv_objdetect 
#    opencv_photo 
#    opencv_stitching
#    opencv_videoio 
#    opencv_video  
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

# c/c++ flags
set(FACE_DETECT_LIBS 
    face_detect
    rknnrt
    ${OpenCV_LIBS} 
    pthread
    stdc++ 
    )
