#pragma once
#include <thread>
#include <vector>
#include <functional>
#include <chrono>
#include <atomic>
#include <string>
#include <exception>
#include <unistd.h>
#include "log.h"

/**
 * @brief 通用任务管理器
 */
class TaskManager {
public:
    struct Task {
        std::string name;                             // 任务名称
        std::function<void()> func;                   // 任务函数
        int niceValue;                                // 线程优先级（越大越低）
        std::chrono::milliseconds interval;           // 执行间隔
        bool repeat;                                  // 是否循环执行
        std::thread thread;                           // 线程对象
        std::atomic<bool> running{false};             // 运行标志
    };

    TaskManager() = default;
    ~TaskManager() {
        stopAll();
    }

    /**
     * @brief 添加任务
     * @param name       任务名
     * @param func       执行函数
     * @param niceValue  优先级
     * @param interval   执行间隔（毫秒）
     * @param repeat     是否循环执行
     */
    void addTask(const std::string& name,
                 std::function<void()> func,
                 int niceValue = 0,
                 std::chrono::milliseconds interval = std::chrono::milliseconds(0),
                 bool repeat = true)
    {
        Task task{name, func, niceValue, interval, repeat};
        tasks.push_back(std::move(task));
    }

    /**
     * @brief 启动所有任务
     */
    void startAll() {
        for (auto &t : tasks) {
            if (t.running) continue;

            t.running = true;
            t.thread = std::thread([&t]() {
                nice(t.niceValue);
                log_info("[TaskManager] Task '%s' started (nice=%d, interval=%lld ms)", 
                         t.name.c_str(), t.niceValue, (long long)t.interval.count());

                if (t.repeat) {
                    while (t.running) {
                        auto start = std::chrono::steady_clock::now();
                        try {
                            t.func();
                        } catch (const std::exception& e) {
                            log_error("[TaskManager] Task '%s' exception: %s", 
                                      t.name.c_str(), e.what());
                        }

                        auto end = std::chrono::steady_clock::now();
                        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                        if (t.interval > elapsed)
                            std::this_thread::sleep_for(t.interval - elapsed);
                    }
                } else {
                    try {
                        t.func();
                    } catch (const std::exception& e) {
                        log_error("[TaskManager] Task '%s' exception: %s", 
                                  t.name.c_str(), e.what());
                    }
                }

                log_info("[TaskManager] Task '%s' stopped.", t.name.c_str());
            });
        }
    }

    /**
     * @brief 停止所有任务并等待退出
     */
    void stopAll() {
        log_info("[TaskManager] Stopping all tasks...");
        for (auto &t : tasks) {
            t.running = false;
        }
        for (auto &t : tasks) {
            if (t.thread.joinable())
                t.thread.join();
        }
        log_info("[TaskManager] All tasks stopped.");
    }

    /**
     * @brief 停止并移除指定任务
     */
    void stopTask(const std::string& name) {
        for (auto &t : tasks) {
            if (t.name == name) {
                log_info("[TaskManager] Stopping task '%s'...", name.c_str());
                t.running = false;
                if (t.thread.joinable())
                    t.thread.join();
                break;
            }
        }
    }

    /**
     * @brief 打印当前任务状态
     */
    void printStatus() const {
        log_info("=== TaskManager Status ===");
        for (const auto &t : tasks) {
            log_info("Task: %s | Running: %s | Interval: %lld ms | Nice: %d",
                     t.name.c_str(),
                     t.running ? "Yes" : "No",
                     (long long)t.interval.count(),
                     t.niceValue);
        }
    }

private:
    std::vector<Task> tasks;
};
