#pragma once
#include "command.h"
#include <mutex>
#include <queue>
#include <functional>
#include <string>
#include <curl/curl.h>
#include <iostream>
extern "C" {
#include "log.h"
}

class CommandManager {
public:
    using CommandCallback = std::function<void(const Command&)>;

    CommandManager(const std::string& eqCode, const std::string& confirmUrl)
        : eqCode(eqCode), confirmUrl(confirmUrl) {}

    void parseServerResponse(const std::string& respJson) {
        try {
            auto j = nlohmann::json::parse(respJson);
            if (!j.contains("command")) return; 
            auto commands = j["command"];
            std::lock_guard<std::mutex> lock(mtx);
            for (auto& c : commands) {
                Command cmd;
                cmd.type = c["type"];
                cmd.index = c["index"];
                cmd.content = c["content"];
                commandQueue.push(cmd);
            }
        } catch (const std::exception& e) {
            log_error("CommandManager: JSON parse error: %s", e.what());
        }
    }

    void setCallback(CommandCallback cb) { callback = cb; }

    void executeCommands() {
        while (!commandQueue.empty()) {
            Command cmd;
            {
                std::lock_guard<std::mutex> lock(mtx);
                cmd = commandQueue.front();
                commandQueue.pop();
            }

            if (callback) callback(cmd); // 执行具体逻辑

            confirmCommand(cmd.index);   // 完成后确认
        }
    }

private:
    std::string eqCode;
    std::string confirmUrl;
    std::queue<Command> commandQueue;
    std::mutex mtx;
    CommandCallback callback;

    void confirmCommand(const std::string& index) {
        CURL* curl = curl_easy_init();
        if (!curl) return;

        curl_mime* mime = curl_mime_init(curl);
        curl_mimepart* part = curl_mime_addpart(mime);
        curl_mime_name(part, "index");
        curl_mime_data(part, index.c_str(), CURL_ZERO_TERMINATED);

        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, ("eq-code: " + eqCode).c_str());

        curl_easy_setopt(curl, CURLOPT_URL, confirmUrl.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);

        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK)
            log_error("CommandManager: confirm failed: %s", curl_easy_strerror(res));
        else    
            log_info("CommandManager: confirm succeeded");

        curl_slist_free_all(headers);
        curl_mime_free(mime);
        curl_easy_cleanup(curl);
    }
};
