#include "heartbeat_task.h"
#include <curl/curl.h>
#include <iostream>
extern "C" {
#include "log.h"
}

HeartbeatTask::HeartbeatTask(const std::string& eqCode,
                             const std::string& url,
                             std::chrono::seconds interval)
    : eqCode(eqCode), url(url), interval(interval), running(false) {}

HeartbeatTask::~HeartbeatTask() {
    stop();
}

void HeartbeatTask::start() {
    if (running) return;
    running = true;
    worker = std::thread(&HeartbeatTask::run, this);
}

void HeartbeatTask::stop() {
    running = false;
    if (worker.joinable())
        worker.join();
}

void HeartbeatTask::setCallback(Callback cb) {
    callback = cb;
}

void HeartbeatTask::updateData(const HeartbeatData& data) {
    hbData = data;
}

void HeartbeatTask::run() {
    log_info("HeartbeatTask: started, interval=%lld sec", (long long)interval.count());
    while (running) {
        auto start = std::chrono::steady_clock::now();
        try {
            sendHeartbeat();
        } catch (const std::exception& e) {
            log_error("HeartbeatTask: exception: %s", e.what());
        } catch (...) {
            log_error("HeartbeatTask: unknown error");
        }

        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start);
        if (interval > elapsed)
            std::this_thread::sleep_for(interval - elapsed);
    }
    log_info("HeartbeatTask: stopped");
}

void HeartbeatTask::sendHeartbeat() {
    if (!running) return;  
    log_debug("HeartbeatTask: sending heartbeat");

    CURL* curl = curl_easy_init();
    if (!curl) {
        log_error("HeartbeatTask: curl init failed");
        return;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, ("eq-code: " + eqCode).c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    curl_mime* mime = curl_mime_init(curl);
    auto addPart = [&](const char* name, const std::string& value){
        auto part = curl_mime_addpart(mime);
        curl_mime_name(part, name);
        curl_mime_data(part, value.c_str(), CURL_ZERO_TERMINATED);
    };

    addPart("time", hbData.time);
    addPart("power", hbData.power);
    addPart("latitude", hbData.latitude);
    addPart("longitude", hbData.longitude);
    addPart("isNorth", std::to_string(hbData.isNorth));
    addPart("isEast", std::to_string(hbData.isEast));
    curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);

    std::string response;
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, +[](char* ptr, size_t size, size_t nmemb, void* userdata) -> size_t {
        auto* resp = reinterpret_cast<std::string*>(userdata);
        resp->append(ptr, size * nmemb);
        return size * nmemb;
    });
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        log_error("HeartbeatTask: curl_easy_perform failed: %s", curl_easy_strerror(res));
    } else {
        log_debug("HeartbeatTask: got response: %s", response.c_str());
        if (callback) {
            try {
                callback(response);
            } catch (const std::exception& e) {
                log_error("HeartbeatTask: callback exception: %s", e.what());
            }
        }
    }

    curl_mime_free(mime);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
}

