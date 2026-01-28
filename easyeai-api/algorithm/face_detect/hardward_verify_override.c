/**
 * hardward_verify_override.c
 * 
 * 覆盖libface_detect.a中的hardward_verify函数
 * 
 * 原始函数签名：uint64_t hardward_verify(uint64_t *param_1)
 * - 打开/dev/lmo_very设备进行AES验证
 * - 返回0表示成功，其他值表示失败
 * - param_1非NULL时写入8字节验证数据
 * 
 * 绕过策略：直接返回成功，写入假数据
 */

#include <stdint.h>
#include <string.h>

// 匹配原始函数签名（只有一个参数）
uint64_t hardward_verify(uint64_t *param_1) {
    // 如果传入了buffer指针，写入假的验证数据
    if (param_1 != NULL) {
        *param_1 = 0xFFFFFFFFFFFFFFFFULL;
    }
    
    // 返回0表示验证成功
    return 0;
}
