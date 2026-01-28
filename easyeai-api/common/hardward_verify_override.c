/**
 * hardward_verify_override.c - 统一的硬件验证绕过
 * 
 * 覆盖libperson_detect.a和libface_detect.a中的hardward_verify函数
 * 两个库的函数签名不同，这里提供两个版本
 */

#include <stdint.h>
#include <string.h>

// person_detect版本：int hardward_verify(uint8_t *buffer, int param)
// 由于符号冲突，我们只定义一个版本，但要能处理两种调用
// 实际上链接器会优先使用.o中的符号，覆盖.a库中的符号

// 注意：虽然两个库的签名不同，但C链接器只看函数名
// 我们需要提供一个兼容两种调用的版本
// 最安全的做法是提供一个通用的返回成功的版本

/**
 * 通用硬件验证绕过函数
 * 
 * person_detect调用: hardward_verify(buffer, param) 返回int
 * face_detect调用: hardward_verify(buffer) 返回uint64_t
 * 
 * 由于C语言函数名不包含参数信息，链接器会看到相同的符号
 * 我们提供一个兼容版本，处理第一个参数为指针的情况
 */
uint64_t hardward_verify(void* param_1, int param_2) {
    // 统一处理：如果第一个参数是有效指针，写入验证数据
    if (param_1 != NULL) {
        // person_detect期望8字节的0xFF
        // face_detect期望一个uint64_t
        uint64_t* ptr = (uint64_t*)param_1;
        *ptr = 0xFFFFFFFFFFFFFFFFULL;
    }
    
    // 返回0表示验证成功（对两个库都适用）
    return 0;
}
