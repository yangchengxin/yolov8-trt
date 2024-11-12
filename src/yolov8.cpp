#include "yolov8.h"
#include "utils.h"

#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line) {
	if (code != cudaSuccess) {
		const char* err_name = cudaGetErrorName(code);
		const char* err_message = cudaGetErrorString(code);
		printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
		return false;
	}
	return true;
}

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

/* 给输入和输出分配内存 */
void YOLOV8::Malloc_data()
{
	int in_out_nums				= _engine->getNbBindings();
	std::cout << "输入和输出的总个数：" << in_out_nums << std::endl;

	const char* name			= _engine->getIOTensorName(0);
	const char* name1			= _engine->getIOTensorName(1);
	const char* name2			= _engine->getIOTensorName(2);
	input_dim					= _engine->getTensorShape(name);
	seg_dim						= _engine->getTensorShape(name1);
	det_dim						= _engine->getTensorShape(name2);

	/* 在cpu和gpu上开辟内存 */
	for (int ii = 0; ii < input_dim.nbDims; ii++)
	{
		input_numel				*= input_dim.d[ii];
	}
	for (int ii = 0; ii < det_dim.nbDims; ii++)
	{
		output_det_numel		*= det_dim.d[ii];
	}
	for (int ii = 0; ii < seg_dim.nbDims; ii++)
	{
		output_seg_numel		*= seg_dim.d[ii];
	}
	checkRuntime(cudaMallocHost(&input_host_data, input_numel * sizeof(float)));
	checkRuntime(cudaMalloc(&input_device_data, input_numel * sizeof(float)));
	checkRuntime(cudaMallocHost(&output_host.det_boxes, output_det_numel * sizeof(float)));
	checkRuntime(cudaMalloc(&output_device.det_boxes, output_det_numel * sizeof(float)));
	checkRuntime(cudaMallocHost(&output_host.seg_protos, output_seg_numel * sizeof(float)));
	checkRuntime(cudaMalloc(&output_device.seg_protos, output_seg_numel * sizeof(float)));
}

/* 初始化模型给输入输出节点分配内存 并返回可推理的上下文 */
void YOLOV8::Initial_model()
{
	nvinfer1::ILogger* gLogger = NULL;
	initLibNvInferPlugins(gLogger, "");

	TRTLogger logger;

	ifstream fs(_model_path, std::ios::binary);
	std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(fs), {});
	fs.close();

	_runtime					= nvinfer1::createInferRuntime(logger);
	_engine						= _runtime->deserializeCudaEngine((void*)buffer.data(), buffer.size());

	checkRuntime(cudaStreamCreate(&_stream));
	auto context				= _engine->createExecutionContext();
	Malloc_data();

	_context					= context;
}

/* 前处理 */
cv::Mat YOLOV8::Preprocess()
{
	_ori_input					= _input;
	float ori_height			= _input.rows;
	float ori_width				= _input.cols;
	float height				= input_dim.d[2];
	float width					= input_dim.d[3];
	scale						= std::min(height / ori_height, width / ori_width);

	const cv::Matx23f matrix
	{
		scale, 0.0, 0.0,
		0.0, scale, 0.0,
	};

	cv::warpAffine(_input, _input, matrix, cv::Size(ori_width * scale, ori_height * scale));
	_input.convertTo(_input, CV_32FC3, 1.0 / 255);
	cv::cvtColor(_input, _input, cv::COLOR_BGR2RGB);

	cv::Mat Pad(cv::Size(input_dim.d[2], input_dim.d[3]), CV_32FC3, cv::Scalar(0, 0, 0));
	cv::Rect roi;
	if (ori_height > ori_width)
	{
		roi						= cv::Rect((input_dim.d[2] - ori_width * scale) / 2, 0, ori_width * scale, ori_height * scale);
		displacement			= input_dim.d[2] - ori_width * scale;
		up = false;
	}
	else
	{
		roi						= cv::Rect(0, (input_dim.d[3] - ori_height * scale) / 2, ori_width * scale, ori_height * scale);
		displacement			= input_dim.d[3] - ori_height * scale;
		up						= true;
	}
	cv::Mat roi_region = Pad(roi);
	_input.copyTo(roi_region);
	return Pad;
}

void YOLOV8::Infer(cv::Mat input)
{
	_input						= input;

	/* 获取类别 */
	const char* class_path		= "cocoNames.txt";
	_classes					= getClassName(class_path);

	/* 对输入图像进行前处理 */
	T->start_cpu();
	cv::Mat transformed_input	= Preprocess();
	T->stop_cpu();
	T->duration_cpu<Timer::ms>("preprocess time:");

	/* 将前处理后的输入填入gpu的输入指针中 */
	int input_cols				= transformed_input.cols;
	int input_rows				= transformed_input.rows;
	int input_channels			= transformed_input.channels();
	std::vector<cv::Mat> chw;
	for (size_t c = 0; c < input_channels; c++)
	{
		chw.emplace_back(cv::Mat(cv::Size(input_cols, input_rows), CV_32FC1, input_host_data + c * input_cols * input_rows));
	}
	cv::split(transformed_input, chw);
	checkRuntime(cudaMemcpy(input_device_data, input_host_data, input_numel * sizeof(float), cudaMemcpyHostToDevice));
	float* buffers[]			= { input_device_data, output_device.seg_protos, output_device.det_boxes };

	nvinfer1::Dims input_dim	= _engine->getBindingDimensions(0);
	_context->setBindingDimensions(0, input_dim);

	/* 执行推理 */
	bool success				= _context->enqueueV2((void**)buffers, _stream, nullptr);

	//同步方式拷贝
	checkRuntime(cudaMemcpy(output_host.det_boxes, output_device.det_boxes, output_det_numel * sizeof(float), cudaMemcpyDeviceToHost));
	checkRuntime(cudaMemcpy(output_host.seg_protos, output_device.seg_protos, output_seg_numel * sizeof(float), cudaMemcpyDeviceToHost));

	/* 将输出解码成mat格式 */
	/* 38 * 8400 -> 8400 * 38 */
	_det						= cv::Mat(det_dim.d[1], det_dim.d[2], CV_32FC1, output_host.det_boxes).t();
	/* 1 * 32 * 160 * 160 -> 1 * 32 * 25600 */
	_seg						= cv::Mat(seg_dim.d[1], seg_dim.d[2] * seg_dim.d[3], CV_32F, output_host.seg_protos);

	T->start_cpu();
	Postprocess();
	T->stop_cpu();
	T->duration_cpu<Timer::ms>("postprocess time:");
}

/* 后处理 */
void YOLOV8::Postprocess()
{
	std::vector<cv::Rect>		mask_boxes;
	std::vector<cv::Rect>		boxes;
	std::vector<int>			class_ids;
	std::vector<float>			confidences;
	std::vector<cv::Mat>		masks;
	for (int i = 0; i < _det.rows; i++)
	{
		/* 获取一个检测中，2个类别的置信度 */
		const cv::Mat	class_scores			= _det.row(i).colRange(4, 6);
		cv::Point		class_id_point;
		double			score;
		cv::minMaxLoc(class_scores, nullptr, &score, nullptr, &class_id_point);

		/* 保留置信度满足阈值的检测 */
		if (score > conf_threshold)
		{
			class_ids.push_back(class_id_point.x);
			confidences.push_back(score);

			/* 从160 -> 640的mask缩放因子 */
			const float		mask_scale			= 0.25f;
			const cv::Mat	detection_box_		= _det.row(i).colRange(0, 4);
			cv::Mat detection_box = cv::Mat(1, 4, CV_32FC1, cv::Scalar(0));
			if (up)
			{
				detection_box.at<float>(0) = detection_box_.at<float>(0);
				detection_box.at<float>(1) = detection_box_.at<float>(1) - displacement / 2;
				detection_box.at<float>(2) = detection_box_.at<float>(2);
				detection_box.at<float>(3) = detection_box_.at<float>(3);
			}
			else
			{
				detection_box.at<float>(0) = detection_box_.at<float>(0) - displacement / 2;
				detection_box.at<float>(1) = detection_box_.at<float>(1);
				detection_box.at<float>(2) = detection_box_.at<float>(2);
				detection_box.at<float>(3) = detection_box_.at<float>(3);
			}
			/* mask在160 * 160尺寸的图上的xyxy */
			const cv::Rect	mask_box			= xywhToxyxy(detection_box_ * mask_scale, cv::Rect(0, 0, 160, 160));
			/* detection在原图上的xywh */
			const cv::Rect	image_box			= xywhToxyxy(detection_box * 1 / scale, cv::Rect(0, 0, _ori_input.cols, _ori_input.rows));
			mask_boxes.push_back(mask_box);
			boxes.push_back(image_box);
			masks.push_back(_det.row(i).colRange(6, 38));
		}
	}


	/* nms */
	std::vector<int> indexes;
	cv::dnn::NMSBoxes(boxes, confidences, 0.25f, 0.45f, indexes);
	std::vector<Segmentblob> segmentblobs;
	for (const int index : indexes)
	{
		Segmentblob segmentblob;
		segmentblob.class_id			= class_ids[index];
		segmentblob.class_confidence	= confidences[index];
		segmentblob.box					= boxes[index];

		cv::Mat m;
		cv::Mat test					= masks[index];
		cv::exp(-masks[index] * _seg, m);
		m								= 1.0f / (1.0f + m);
		m								= m.reshape(1, 160);
		cv::resize(m(mask_boxes[index]) > 0.5f, segmentblob.mask, segmentblob.box.size());
		segmentblobs.push_back(segmentblob);
	}
	//PutMask(_ori_input, segmentblobs, _classes);

	/*cv::namedWindow("seg", cv::WINDOW_NORMAL);
	cv::imshow("seg", _ori_input);
	cv::waitKey(0);*/
	std::cout << "debug" << std::endl;
}
