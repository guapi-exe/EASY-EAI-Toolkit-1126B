#include <atomic>
#include <thread>
#include <chrono>
#include <string>
#include <cstdio>
#include <cstdlib>
extern "C" {
#include "log.h"
#include "gpio.h"
}

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

class GPIOMonitorTask {
public:
    GPIOMonitorTask(const std::string& pinName)
        : pinName_(pinName), running_(false), gpioInitialized_(false) {}

    ~GPIOMonitorTask() { stop(); }

    static void initGPIO(const GPIOCfg_t* cfg, size_t cfgSize) {
        gpio_init(cfg, cfgSize);
    }

    void defaultInitGPIO() {
        if (gpioInitialized_) return;
        static const GPIOCfg_t gpioCfg_tab[] = {
            {
                .pinName   = "GPIO5_C0",
                .direction = DIR_OUTPUT,
                .val       = 0,
            }, {
                .pinName   = "GPIO5_C1",
                .direction = DIR_INPUT,
                .val       = 0,
            }
        };
        gpio_init(gpioCfg_tab, ARRAY_SIZE(gpioCfg_tab));
        gpioInitialized_ = true;
    }

    void start(bool useDefaultInit = false) {
        if (running_) return;
        if (useDefaultInit) defaultInitGPIO();
        running_ = true;
        worker_ = std::thread(&GPIOMonitorTask::run, this);
    }

    void stop() {
        running_ = false;
        if (worker_.joinable()) worker_.join();
    }

private:
    void run() {
        const int sleepThreshold = 120; // 2分钟，单位秒
        int lowCount = 0;
        while (running_) {
            int val = read_pin_val(pinName_.c_str());
            if (val == 0) {
                lowCount++;
                if (lowCount >= sleepThreshold) {
                    log_debug("系统休眠\n", pinName_.c_str());
                    system("echo freeze > /sys/power/state");
                    break;
                }
            } else {
                lowCount = 0;
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }

    std::string pinName_;
    std::atomic<bool> running_;
    std::thread worker_;
    bool gpioInitialized_;
};