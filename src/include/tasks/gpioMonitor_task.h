
#pragma once
#include <string>
#include <thread>
#include <atomic>

extern "C" {
#include "gpio.h"
}

class GPIOMonitorTask {
public:
	GPIOMonitorTask(const std::string& pinName);
	~GPIOMonitorTask();

	static void initGPIO(const GPIOCfg_t* cfg, size_t cfgSize);
	void defaultInitGPIO();

	void start(bool useDefaultInit = false);
	void stop();

private:
	void run();

	std::string pinName_;
	std::atomic<bool> running_;
	std::thread worker_;
	bool gpioInitialized_;
};
