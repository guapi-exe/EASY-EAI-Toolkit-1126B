#pragma once
#include <string>
#include <functional>
#include <vector>
#include <unordered_map>
#include <nlohmann/json.hpp>

struct Command {
    int type;                   // 指令类型
    std::string index;           // 指令索引
    nlohmann::json content;      // 指令上下文
};
