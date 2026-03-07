#include "tcp_client.h"

#include "json.hpp"
extern "C" {
#include "log.h"
}

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

using nlohmann::json;

namespace {
constexpr size_t kMaxPendingMessages = 1000;
}

TcpClient::TcpClient(DeviceConfig* cfg, const std::string& cfgPath)
    : config(cfg), configPath(cfgPath) {}

TcpClient::~TcpClient() {
    stop();
}

void TcpClient::start() {
    if (running) return;
    running = true;
    worker = std::thread(&TcpClient::workerLoop, this);
    heartbeatWorker = std::thread(&TcpClient::heartbeatLoop, this);
}

void TcpClient::stop() {
    running = false;
    closeSocket();
    if (worker.joinable()) worker.join();
    if (heartbeatWorker.joinable()) heartbeatWorker.join();
}

void TcpClient::setCommandCallback(CommandCallback cb) {
    std::lock_guard<std::mutex> lock(cbMtx);
    commandCallback = cb;
}

void TcpClient::sendPersonAppeared(int personId) {
    json j = {
        {"type", "event"},
        {"event", "person_appeared"},
        {"person_id", personId},
        {"ts", (long long)time(nullptr)}
    };
    queueJsonLine(j.dump());
}

void TcpClient::sendCaptureComplete(int personId, const std::string& imageType) {
    json j = {
        {"type", "event"},
        {"event", "capture_complete"},
        {"person_id", personId},
        {"image_type", imageType},
        {"ts", (long long)time(nullptr)}
    };
    queueJsonLine(j.dump());
}

void TcpClient::queueJsonLine(const std::string& jsonLine) {
    std::lock_guard<std::mutex> lock(sendQueueMtx);
    if (sendQueue.size() >= kMaxPendingMessages) {
        sendQueue.pop();
    }
    sendQueue.push(jsonLine + "\n");
}

void TcpClient::tryConnect() {
    if (connected) return;

    int localSock = socket(AF_INET, SOCK_STREAM, 0);
    if (localSock < 0) {
        return;
    }

    sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(config->tcpPort);
    if (inet_pton(AF_INET, config->tcpServerIp.c_str(), &addr.sin_addr) <= 0) {
        close(localSock);
        return;
    }

    if (connect(localSock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        close(localSock);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(ioMtx);
        sockfd = localSock;
        connected = true;
        recvBuffer.clear();
    }

    json reg = {
        {"type", "register"},
        {"device_code", config->deviceCode},
        {"ts", (long long)time(nullptr)}
    };
    queueJsonLine(reg.dump());

    log_info("TcpClient: connected to %s:%d", config->tcpServerIp.c_str(), config->tcpPort);
}

void TcpClient::closeSocket() {
    std::lock_guard<std::mutex> lock(ioMtx);
    if (sockfd >= 0) {
        close(sockfd);
        sockfd = -1;
    }
    connected = false;
}

void TcpClient::flushSendQueue() {
    if (!connected) return;

    std::queue<std::string> pending;
    {
        std::lock_guard<std::mutex> lock(sendQueueMtx);
        std::swap(pending, sendQueue);
    }

    while (!pending.empty() && connected) {
        std::string msg = pending.front();
        pending.pop();

        int fdCopy = -1;
        {
            std::lock_guard<std::mutex> lock(ioMtx);
            fdCopy = sockfd;
        }

        if (fdCopy < 0) {
            connected = false;
            break;
        }

        ssize_t sent = send(fdCopy, msg.data(), msg.size(), MSG_NOSIGNAL);
        if (sent < 0 || (size_t)sent != msg.size()) {
            connected = false;
            closeSocket();
            break;
        }
    }

    if (!pending.empty()) {
        std::lock_guard<std::mutex> lock(sendQueueMtx);
        while (!pending.empty()) {
            sendQueue.push(pending.front());
            pending.pop();
        }
    }
}

void TcpClient::handleIncomingBuffer(const std::string& incoming) {
    recvBuffer += incoming;

    size_t pos = std::string::npos;
    while ((pos = recvBuffer.find('\n')) != std::string::npos) {
        std::string line = recvBuffer.substr(0, pos);
        recvBuffer.erase(0, pos + 1);
        if (!line.empty()) {
            handleMessage(line);
        }
    }
}

void TcpClient::handleMessage(const std::string& line) {
    try {
        json j = json::parse(line);
        std::string type = j.value("type", "");

        if (type == "config_update") {
            if (config->applyServerConfig(line)) {
                config->save(configPath);
                log_info("TcpClient: applied server config update");
            }

            CommandCallback cb;
            {
                std::lock_guard<std::mutex> lock(cbMtx);
                cb = commandCallback;
            }
            if (cb) {
                cb("config_update", line);
            }
            return;
        }

        std::string cmd = j.value("cmd", "");
        if (!cmd.empty()) {
            CommandCallback cb;
            {
                std::lock_guard<std::mutex> lock(cbMtx);
                cb = commandCallback;
            }
            if (cb) {
                cb(cmd, line);
            }
        }
    } catch (...) {
        log_warn("TcpClient: invalid message: %s", line.c_str());
    }
}

void TcpClient::workerLoop() {
    while (running) {
        if (!connected) {
            tryConnect();
            if (!connected) {
                std::this_thread::sleep_for(std::chrono::seconds(config->reconnectIntervalSec));
                continue;
            }
        }

        flushSendQueue();

        int fdCopy = -1;
        {
            std::lock_guard<std::mutex> lock(ioMtx);
            fdCopy = sockfd;
        }
        if (fdCopy < 0) {
            connected = false;
            continue;
        }

        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(fdCopy, &readfds);
        timeval tv;
        tv.tv_sec = 1;
        tv.tv_usec = 0;

        int sel = select(fdCopy + 1, &readfds, nullptr, nullptr, &tv);
        if (sel < 0) {
            closeSocket();
            continue;
        }
        if (sel == 0) {
            continue;
        }

        if (FD_ISSET(fdCopy, &readfds)) {
            char buf[1024];
            ssize_t n = recv(fdCopy, buf, sizeof(buf), 0);
            if (n <= 0) {
                closeSocket();
                continue;
            }
            handleIncomingBuffer(std::string(buf, buf + n));
        }
    }
}

void TcpClient::heartbeatLoop() {
    while (running) {
        if (connected) {
            queueJsonLine(buildHeartbeatJson());
        }
        std::this_thread::sleep_for(std::chrono::seconds(config->heartbeatIntervalSec));
    }
}

std::string TcpClient::buildHeartbeatJson() const {
    json j = {
        {"type", "heartbeat"},
        {"device_code", config->deviceCode},
        {"cpu_temp_c", readCpuTemperatureC()},
        {"cpu_usage", readCpuUsagePercent()},
        {"mem_usage", readMemoryUsagePercent()},
        {"ts", (long long)time(nullptr)}
    };
    return j.dump();
}

double TcpClient::readCpuTemperatureC() {
    std::ifstream ifs("/sys/class/thermal/thermal_zone0/temp");
    if (!ifs.is_open()) return 0.0;
    double milli = 0.0;
    ifs >> milli;
    return milli / 1000.0;
}

double TcpClient::readMemoryUsagePercent() {
    std::ifstream ifs("/proc/meminfo");
    if (!ifs.is_open()) return 0.0;

    std::string key;
    double value = 0;
    std::string unit;
    double total = 0;
    double available = 0;

    while (ifs >> key >> value >> unit) {
        if (key == "MemTotal:") total = value;
        if (key == "MemAvailable:") {
            available = value;
            break;
        }
    }

    if (total <= 0) return 0.0;
    return (total - available) * 100.0 / total;
}

double TcpClient::readCpuUsagePercent() {
    static unsigned long long lastUser = 0;
    static unsigned long long lastNice = 0;
    static unsigned long long lastSystem = 0;
    static unsigned long long lastIdle = 0;

    std::ifstream ifs("/proc/stat");
    if (!ifs.is_open()) return 0.0;

    std::string cpu;
    unsigned long long user = 0, nice = 0, system = 0, idle = 0;
    ifs >> cpu >> user >> nice >> system >> idle;

    unsigned long long userDiff = user - lastUser;
    unsigned long long niceDiff = nice - lastNice;
    unsigned long long systemDiff = system - lastSystem;
    unsigned long long idleDiff = idle - lastIdle;

    lastUser = user;
    lastNice = nice;
    lastSystem = system;
    lastIdle = idle;

    unsigned long long total = userDiff + niceDiff + systemDiff + idleDiff;
    if (total == 0) return 0.0;

    return (double)(userDiff + niceDiff + systemDiff) * 100.0 / (double)total;
}
