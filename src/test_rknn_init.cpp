#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>

extern "C" {
#include "log.h"
}

#include "person_detect.h"
#include "face_detect.h"

int main(int argc, char** argv) {
    const char* person_model = "person_detect.model";
    const char* face_model = "face_detect.model";
    
    if (argc > 1) {
        person_model = argv[1];
    }
    if (argc > 2) {
        face_model = argv[2];
    }
    
    log_info("==== RKNN 模型加载测试 ====");
    log_info("Person Model: %s", person_model);
    log_info("Face Model: %s", face_model);
    log_info("Working Dir: %s", getcwd(NULL, 0));
    
    // 检查 NPU 设备
    log_info("Checking NPU device...");
    if (access("/dev/rknpu", F_OK) == 0) {
        log_info("  /dev/rknpu exists");
    } else {
        log_warn("  /dev/rknpu NOT found");
    }
    
    // 检查模型文件
    log_info("Checking model files...");
    if (access(person_model, F_OK) == 0) {
        log_info("  %s exists", person_model);
    } else {
        log_error("  %s NOT found", person_model);
        return -1;
    }
    
    if (access(face_model, F_OK) == 0) {
        log_info("  %s exists", face_model);
    } else {
        log_error("  %s NOT found", face_model);
        return -1;
    }
    
    log_info("");
    log_info("=== Starting person_detect_init ===");
    fflush(stdout);
    fflush(stderr);
    
    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    rknn_context personCtx;
    int ret = person_detect_init(&personCtx, person_model);
    
    gettimeofday(&end, NULL);
    long elapsed_ms = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000;
    
    if (ret != 0) {
        log_error("person_detect_init FAILED: ret=%d, time=%ld ms", ret, elapsed_ms);
        return -1;
    }
    
    log_info("person_detect_init SUCCESS: time=%ld ms", elapsed_ms);
    log_info("");
    log_info("=== Starting face_detect_init ===");
    fflush(stdout);
    fflush(stderr);
    
    gettimeofday(&start, NULL);
    
    rknn_context faceCtx;
    ret = face_detect_init(&faceCtx, face_model);
    
    gettimeofday(&end, NULL);
    elapsed_ms = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000;
    
    if (ret != 0) {
        log_error("face_detect_init FAILED: ret=%d, time=%ld ms", ret, elapsed_ms);
        person_detect_release(personCtx);
        return -1;
    }
    
    log_info("face_detect_init SUCCESS: time=%ld ms", elapsed_ms);
    log_info("");
    log_info("=== Releasing models ===");
    
    person_detect_release(personCtx);
    face_detect_release(faceCtx);
    
    log_info("All tests passed!");
    log_info("==========================");
    
    return 0;
}
