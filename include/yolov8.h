#pragma once
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <chrono>
#include <fstream>
#include <windows.h>
#include <vector>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <NvInferPlugin.h>
#include <assert.h>
#include <math.h>
#include "utils.h"

using namespace std;
using namespace nvinfer1;

struct Output_ptr
{
	float* det_boxes;
	float* seg_protos;
};

class YOLOV8 : public Utils
{
public:
	YOLOV8(const char* model_path) : _model_path(model_path)
	{
		
	}
	~YOLOV8()
	{
		delete T;
		
	}

	void Initial_model();
	void Malloc_data();
	cv::Mat Preprocess();
	void Postprocess();
	void Infer(cv::Mat input);

	cv::Mat show;

private:
	Timer* T = new Timer();

	float scale;
	const char*						_model_path;
	cv::Mat							_ori_input;
	cv::Mat							_input;
	cv::Mat							_det;
	cv::Mat							_seg;

	cudaStream_t					_stream						= nullptr;
	nvinfer1::ICudaEngine*			_engine						= nullptr;
	nvinfer1::IRuntime*				_runtime					= nullptr;
	nvinfer1::IExecutionContext*	_context					= nullptr;

	/* 初始化输入和输出指针 */
	float*							input_host_data				= nullptr;
	float*							input_device_data			= nullptr;
	Output_ptr						output_host					= { nullptr, nullptr };
	Output_ptr						output_device				= { nullptr, nullptr };

	nvinfer1::Dims					input_dim;
	nvinfer1::Dims					det_dim;
	nvinfer1::Dims					seg_dim;

	bool							up							= true;
	int								displacement				= 0;
	int								input_numel					= 1;
	int								output_det_numel			= 1;
	int								output_seg_numel			= 1;
	const float						conf_threshold				= 0.2;
	const float						nms_threshold				= 0.5;

	std::vector<std::string>		_classes;
};