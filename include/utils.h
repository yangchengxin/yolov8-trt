#pragma once
#include <iostream>
#include <fstream>
#include <time.h>
#include <io.h>
#include <direct.h>
#include <chrono>
#include <system_error>
#include <stdarg.h>
#include <opencv2/opencv.hpp>
#define LOG(...)                     __log_info(__VA_ARGS__)

using namespace std;

struct Segmentblob
{
    int			class_id;
    float		class_confidence;
    cv::Rect2f	box;
    cv::Mat		mask;
};

static void __log_info(const char* format, ...) {
    char msg[1000];
    va_list args;
    va_start(args, format);

    vsnprintf(msg, sizeof(msg), format, args);

    fprintf(stdout, "%s\n", msg);
    va_end(args);
}

class Utils
{
public:
    Utils() {};
    ~Utils() {};

    cv::Rect xywhToxyxy(const cv::Mat& input, const cv::Rect& range);
    std::vector<std::string> getClassName(std::string class_file);
    void PutMask(cv::Mat& image, std::vector<Segmentblob>& results, std::vector<std::string> class_names);
};

class Timer
{
public:
    using s = std::ratio<1, 1>;
    using ms = std::ratio<1, 1000>;
    using us = std::ratio<1, 1000000>;
    using ns = std::ratio<1, 1000000000>;

    Timer();
    ~Timer();

public:
    void start_cpu();
    void stop_cpu();

    template <typename span>
    void duration_cpu(std::string msg);

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> _cStart;
    std::chrono::time_point<std::chrono::high_resolution_clock> _cStop;
};

template <typename span>
void Timer::duration_cpu(std::string msg) {
    std::string str;

    if (std::is_same<span, s>::value) { str = "s"; }
    else if (std::is_same<span, ms>::value) { str = "ms"; }
    else if (std::is_same<span, us>::value) { str = "us"; }
    else if (std::is_same<span, ns>::value) { str = "ns"; }

    std::chrono::duration<double, span> time = _cStop - _cStart;
    LOG("%-35s uses %.6lf %s", msg.c_str(), time.count(), str.c_str());
}



