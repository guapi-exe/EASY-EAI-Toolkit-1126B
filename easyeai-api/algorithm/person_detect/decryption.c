/**
 * decryption.c - 解密模块（绕过硬件验证版本）
 * 
 * 用于替换 libperson_detect.a 中的 decryption.c.o
 * 移除硬件验证无限循环，支持未加密的RKNN模型
 */

#include <string.h>
#include <stdint.h>

// 全局初始化标志
static int first_init = 0;

/**
 * 初始化解密模块 - 绕过硬件验证
 * 
 * @param param_1 原本用于位掩码检查的参数（已忽略）
 * @return 0 表示成功
 */
int decrypte_init(uint16_t param_1) {
    first_init = 1;
    return 0;
}

/**
 * 模型解密函数 - 直接复制数据（假设未加密）
 * 
 * @param param_1 输入数据指针
 * @param param_2 输出数据指针
 * @param param_3 数据总长度
 * @return 0 表示成功，-1 表示参数错误
 */
int decrypte_model(const void *param_1, void *param_2, int param_3) {
    if (param_1 == NULL || param_2 == NULL || param_3 <= 4) {
        return -1;
    }
    
    // 跳过前4字节（可能是加密头），复制剩余数据
    const uint8_t *src = (const uint8_t *)param_1 + 4;
    uint8_t *dst = (uint8_t *)param_2;
    int copy_size = param_3 - 4;
    
    if (copy_size > 0) {
        memcpy(dst, src, copy_size);
    }
    
    return 0;
}

/**
 * 硬件验证函数桩 - 总是返回成功
 * 
 * @param buffer 输出缓冲区
 * @param param 参数
 * @return 0 表示验证成功
 */
int hardward_verify(uint8_t *buffer, int param) {
    if (buffer != NULL) {
        memset(buffer, 0xFF, 8);
    }
    return 0;
}
