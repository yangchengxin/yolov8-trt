#pragma once
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <fstream>
#include <windows.h>
#include <vector>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <NvInferPlugin.h>
#include <assert.h>
#include <math.h>

using namespace std;
using namespace nvinfer1;


inline const char* severity_string(nvinfer1::ILogger::Severity t) {
	switch (t) {
	case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
	case nvinfer1::ILogger::Severity::kERROR:   return "error";
	case nvinfer1::ILogger::Severity::kWARNING: return "warning";
	case nvinfer1::ILogger::Severity::kINFO:    return "info";
	case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
	default: return "unknow";
	}
}

class TRTLogger : public nvinfer1::ILogger {
public:
	virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
		if (severity <= Severity::kWARNING) {
			// 打印带颜色的字符，格式如下：
			// printf("\033[47;33m打印的文本\033[0m");
			// 其中 \033[ 是起始标记
			//      47    是背景颜色
			//      ;     分隔符
			//      33    文字颜色
			//      m     开始标记结束
			//      \033[0m 是终止标记
			// 其中背景颜色或者文字颜色可不写
			// 部分颜色代码 https://blog.csdn.net/ericbar/article/details/79652086
			if (severity == Severity::kWARNING) {
				printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
			}
			else if (severity <= Severity::kERROR) {
				printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
			}
			else {
				printf("%s: %s\n", severity_string(severity), msg);
			}
		}
	}
};

// 通过智能指针管理nv返回的指针参数
// 内存自动释放，避免泄漏
template<typename _T>
shared_ptr<_T> make_nvshared(_T* ptr) {
	return shared_ptr<_T>(ptr, [](_T* p) {p->destroy(); });
}

int onnx_to_engin(const char* onnx_model, const char* engine_model);
bool build_model();