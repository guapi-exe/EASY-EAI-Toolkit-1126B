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
 * 模型解密函数 - 字节序翻转解密
 * 
 * 解密逻辑：
 * 1. 第0字节是块大小（block_size）
 * 2. 前4字节是头部，数据从第4字节开始
 * 3. 将数据按块分割，每块内字节倒序
 * 4. 不足一块的剩余字节直接复制
 * 
 * @param param_1 输入加密数据指针
 * @param param_2 输出解密数据指针
 * @param param_3 数据总长度
 * @return 0 表示成功，-1 表示参数错误
 */
int decrypte_model(const void *param_1, void *param_2, int param_3) {
    if (param_1 == NULL || param_2 == NULL || param_3 <= 4) {
        return -1;
    }
    
    const uint8_t *src = (const uint8_t *)param_1;
    uint8_t *dest = (uint8_t *)param_2;
    
    // 读取块大小（第0字节）
    uint8_t block_size = src[0];
    int step = (int)block_size + 1;
    
    // 计算数据长度（跳过前4字节头部）
    int data_len = param_3 - 4;
    const uint8_t *data_ptr = src + 4;
    
    // 计算完整块数量和剩余字节
    int num_blocks = (step != 0) ? (data_len / step) : 0;
    int remaining_bytes = (step != 0) ? (data_len % step) : data_len;
    
    int out_idx = 0;
    
    // 逐块处理，每块进行字节序翻转
    for (int b = 0; b < num_blocks; b++) {
        for (int i = 0; i < step; i++) {
            // 将源块的末尾字节放入目标块的起始位置
            dest[out_idx + i] = data_ptr[out_idx + (block_size - i)];
        }
        out_idx += step;
    }
    
    // 处理不足一个完整块的剩余字节（直接拷贝，不翻转）
    for (int i = 0; i < remaining_bytes; i++) {
        dest[out_idx + i] = data_ptr[out_idx + i];
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
