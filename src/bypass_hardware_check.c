/**
 * 硬件校验绕过
 * 
 * RKNN 库中的 decrypte_init 会调用 hardward_verify 进行硬件校验
 * 如果校验失败会进入死循环导致程序卡死
 * 
 * 这个文件提供了 hardward_verify 的替代实现，返回成功状态
 * 链接器会优先使用这个实现而不是库中的实现
 */

#include <string.h>

/**
 * 伪造的硬件校验函数 - 永远返回成功
 * 
 * @param local_10 8字节的硬件ID缓冲区（会被填充）
 * @param param_2  未使用的参数
 * @return 0 表示校验通过
 */
int hardward_verify(unsigned char *local_10, int param_2) {
    // 填充硬件ID为 0xFF（允许所有位通过校验）
    // decrypte_init 会检查特定的位标志
    if (local_10 != NULL) {
        memset(local_10, 0xFF, 8);
    }
    
    // 返回 0 表示校验成功
    return 0;
}
