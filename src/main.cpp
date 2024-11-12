#include "yolov8.h"
#include "onnx2trt.h"
#include "utils.h"

int main(int argc, char** argv)
{
	try
	{
#ifdef ONNX2TRT
		const char* onnx_path = "best.onnx";
		const char* engine_path = "best.engine";
		onnx_to_engin(onnx_path, engine_path);
		/*bool a = build_model();*/
		return 0;
		system("pause");
#endif

#ifdef INFERENCE
		const char* model_path = R"(D:\company_Tenda\6.yolov8.trt\yolov8\yolov8\best.engine)";
		YOLOV8* yolov8 = new YOLOV8(model_path);
		yolov8->Initial_model();

		/* 读取图片 */
		const char* image_path = R"(D:\BaiduNetdiskDownload\362_Image16_original.jpg)";
		cv::Mat input = cv::imread(image_path);

		///* 读取视频 */
		//std::string videopath = R"(D:\BaiduNetdiskDownload\201903281533270065VAS_20241111_105558.avi)";
		//cv::VideoCapture cap(videopath);
		//if (!cap.isOpened())
		//{
		//	std::cerr << "Error: Could not open video file." << std::endl;
		//	return -1;
		//}
		//double fps = cap.get(cv::CAP_PROP_FPS);
		//std::cout << "Frames per second: " << fps << std::endl;
		//cv::Mat frame;

		/* 推理 */
		while (true)
		{
			//cap >> frame;
			//if (frame.empty())
			//{
			//	std::cout << "视频读取完毕！" << std::endl;
			//	break;
			//}
			Timer timer;
			timer.start_cpu();
			yolov8->Infer(input);
			timer.stop_cpu();
			timer.duration_cpu<Timer::ms>("total time:");
		}

		delete yolov8;

#endif // INFERENCE
	}
	catch (const std::exception& e)
	{
		std::cerr << "exception: " << e.what() << std::endl;
	}

	catch (...)
	{
		std::cerr << "unknown exception" << std::endl;
	}
	
	return 0;
	system("pause");
}

