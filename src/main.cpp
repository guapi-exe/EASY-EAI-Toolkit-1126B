#include "task_manager.h"
#include "camera_task.h"
#include "uploader_task.h"
#include "heartbeat_task.h"
#include "command_manager.h"

int main() {
    HeartbeatData hbData; //虚拟信息
    hbData.time = "1696000000";
    hbData.power = "100";
    hbData.latitude = "39.9042";
    hbData.longitude = "116.4074";
    hbData.isNorth = 1;
    hbData.isEast = 1;

    UploaderTask uploader("111", "http://101.200.56.225:11100/receive/image/auto");
    HeartbeatTask heartbeat("111", "http://101.200.56.225:11100/receive/heartbeat", std::chrono::seconds(30));
    CommandManager commandManager("111", "http://101.200.56.225:11100/receive/command/confirm");
    uploader.start();
    commandManager.setCallback([&](const Command& cmd){
        switch(cmd.type) {
            case 1: // 主动抓拍
                log_debug("CommandManager: Capture command received, index=%s", cmd.content["active"].get<int>());
                break;
            case 2: // 修改服务端地址
                log_debug("CommandManager: Set server URL to %s", cmd.content["ip"].get<std::string>().c_str());
                break;
            case 3: // 修改发送间隔
                log_debug("CommandManager: Set heartbeat interval to %d seconds", cmd.content["interval"].get<int>());
                break;
            default:
                break;
        }
    });
    heartbeat.updateData(hbData);
    heartbeat.setCallback([&](const std::string& respJson){
        commandManager.parseServerResponse(respJson);
        commandManager.executeCommands();
    });

    heartbeat.start();
    CameraTask camera("person_detect.model", "face_detect.model", 22);
    camera.setUploadCallback([&](const cv::Mat& img, int id, const std::string& type){
        uploader.enqueue(img, 1, type);
    });

    TaskManager tm;
    tm.addTask("CameraTask", [&](){ camera.start(); }, -5, std::chrono::seconds(1), true);

    tm.startAll();

    std::atomic<bool> running(true);
    signal(SIGINT, [](int){ running = false; });

    log_info("System started. Press Ctrl+C to stop.");

    while (running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    log_info("System shutting down...");
    return 0;
}
