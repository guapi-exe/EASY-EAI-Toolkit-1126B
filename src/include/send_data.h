#ifndef SEND_DATA_H
#define SEND_DATA_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <ctime>

std::string image_to_base64(const cv::Mat& img, const std::string& ext);
std::string get_current_time_string();
std::string build_json(const cv::Mat& img, int id, const std::string& type);

#endif