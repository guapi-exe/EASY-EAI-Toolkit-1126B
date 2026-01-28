/**
 * hardward_verify_override.c
 * 
 * 覆盖libperson_detect.a中的hardward_verify函数
 * 让硬件验证始终通过，绕过无限循环陷阱
 */

#include <stdint.h>
#include <string.h>

/**
 * 硬件验证函数 - 强制返回成功
 * 
 * 这个函数会覆盖.a库中的同名函数
 * 在链接时，目标文件的符号优先级高于静态库
 * 
 * @param buffer 输出缓冲区（8字节）
 * @param param 参数（未使用）
 * @return 0 表示验证成功
 */
int hardward_verify(uint8_t *buffer, int param) {
    // 填充全0xFF表示"所有功能都可用"
    if (buffer != NULL) {
        memset(buffer, 0xFF, 8);
    }
    
    // 总是返回成功，绕过硬件验证
    return 0;
}
