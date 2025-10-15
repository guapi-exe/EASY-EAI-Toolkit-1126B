#include "main.h"
#include "task_manager.h"
#include "camera_task.h"
#include "uploader_task.h"
#include "serial_task.h"
#include "heartbeat_task.h"
#include "command_manager.h"
#include <csignal> 
#include <unistd.h>
#include <sys/select.h>

std::atomic<bool> running(true);

void handleSignal(int) {
    running = false;
}

int main() {
    HeartbeatData hbData; //虚拟信息
    hbData.time = "1696000000";
    hbData.power = "100";
    hbData.latitude = "39.9042";
    hbData.longitude = "116.4074";
    hbData.isNorth = 1;
    hbData.isEast = 1;

    UploaderTask uploader("111", "http://101.200.56.225:11100");
    HeartbeatTask heartbeat("111", "http://101.200.56.225:11100/receive/heartbeat", std::chrono::seconds(30));
    CommandManager commandManager("111", "http://101.200.56.225:11100/receive/command/confirm");
    SerialTask serial("/dev/ttyS2", 115200);
    uploader.start();
    
    heartbeat.updateData(hbData);
    heartbeat.setCallback([&](const std::string& respJson){
        commandManager.parseServerResponse(respJson);
        commandManager.executeCommands();
    });

    CameraTask camera("person_detect.model", "face_detect.model", CAMERA_INDEX_2);
    camera.setUploadCallback([&](const cv::Mat& img, int id, const std::string& type){
        if (type == "all")
        {
            uploader.enqueue(img, 1, type, "/receive/image/auto");
        }else{
            uploader.enqueue(img, 1, type, "/receive/image/auto");
        }
    });

    commandManager.setCallback([&](const Command& cmd){
        switch(cmd.type) {
            case 1: // 主动抓拍
                log_debug("CommandManager: Capture command received, index=%s", cmd.content["active"].get<int>());
                camera.captureSnapshot();
                break;
            case 2: // 修改服务端地址
                log_debug("CommandManager: Set server URL to %s", cmd.content["ip"].get<std::string>().c_str());
                break;
            case 3: // 修改发送间隔
                log_debug("CommandManager: Set heartbeat interval to %d seconds", cmd.content["interval"].get<int>());
                heartbeat.updateInterval(std::chrono::seconds(cmd.content["interval"].get<int>()));
                break;
            default:
                break;
        }
    });
    
    TaskManager tm;
    tm.addTask("CameraTask", [&](){ camera.start(); }, -5, std::chrono::seconds(1), true);
    tm.addTask("HeartbeatTask", [&](){ heartbeat.start(); }, 10, std::chrono::seconds(1), true);
    tm.addTask("SerialTask", [&](){ serial.start(); }, 10, std::chrono::seconds(1), true);
    tm.startAll();

    std::signal(SIGINT, handleSignal);

    log_info("System started. Press Ctrl+C to stop.");

    while (running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));

        fd_set set;
        struct timeval timeout;
        FD_ZERO(&set);
        FD_SET(STDIN_FILENO, &set);
        timeout.tv_sec = 0;
        timeout.tv_usec = 0;
        int rv = select(STDIN_FILENO + 1, &set, NULL, NULL, &timeout);

        if (rv > 0 && FD_ISSET(STDIN_FILENO, &set)) {
            char buf[256] = {0};
            ssize_t len = read(STDIN_FILENO, buf, sizeof(buf) - 1);
            if (len > 0) {
                buf[len] = '\0';
                serial.send(buf);
                log_info("Main: Sent to serial: %s", buf);
            }
        }
    }

    log_info("System shutting down...");
    return 0;
}
