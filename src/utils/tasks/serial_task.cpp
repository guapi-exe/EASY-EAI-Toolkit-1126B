#include "serial_task.h"
#include <thread>
#include <atomic>
#include <string>
#include <vector>
#include <unistd.h>
extern "C" {
#include "log.h"
#include "uart.h"
}

class SerialTask {
public:
    SerialTask(const std::string& port, int baudrate)
        : port_(port), baudrate_(baudrate), running_(false), fd_(-1) {}

    ~SerialTask() { stop(); }

    void start() {
        if (running_) return;
        running_ = true;
        worker_ = std::thread(&SerialTask::run, this);
    }

    void send(const std::string& data) {
        if (running_) {
            UART_Send(fd_, const_cast<char*>(data.c_str()), data.size());
        }
    }

    void stop() {
        running_ = false;
        if (worker_.joinable()) worker_.join();
        if (fd_ >= 0) UART_Close(fd_);
    }

private:
    void run() {
        fd_ = UART_Open(port_.c_str());
        if (fd_ < 0) {
            log_error("SerialTask: Failed to open port %s", port_.c_str());
            return;
        }
        if (!UART_Set(fd_, baudrate_, 0, 8, 1, 'N')) {
            log_error("SerialTask: Failed to set port %s", port_.c_str());
            UART_Close(fd_);
            fd_ = -1;
            return;
        }
        log_info("SerialTask: Serial port %s opened", port_.c_str());

        char recv_buf[256];
        while (running_) {
            int len = UART_Recv(fd_, recv_buf, sizeof(recv_buf));
            if (len > 0) {
                std::string msg(recv_buf, len);
                log_debug("SerialTask: Received: %s", msg.c_str());
            }
            // 可选：发送测试数据
            // UART_Send(fd_, (char*)"hello", 5);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    std::string port_;
    int baudrate_;
    std::atomic<bool> running_;
    std::thread worker_;
    int fd_;
};