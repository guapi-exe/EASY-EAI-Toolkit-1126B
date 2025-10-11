#include "task_manager.h"
#include "camera_task.h"
#include "uploader_task.h"

int main() {
    UploaderTask uploader("http://101.200.56.225:11100/receive/image/auto");
    uploader.start();

    CameraTask camera("person_model.model", "face_model.model", 22);
    camera.setUploadCallback([&](const cv::Mat& img, int id, const std::string& type){
        uploader.enqueue(img, 1, type);
    });

    TaskManager tm;
    tm.addTask("CameraTask", [&](){ camera.start(); }, -5, std::chrono::seconds(1), true);

    tm.startAll();

    std::this_thread::sleep_for(std::chrono::minutes(10));

    camera.stop();
    uploader.stop();
    tm.stopAll();
    return 0;
}
