#include "utils.h"

Timer::Timer() {
	_cStart = std::chrono::high_resolution_clock::now();
	_cStop = std::chrono::high_resolution_clock::now();
}

Timer::~Timer() {}

void Timer::start_cpu() {
	_cStart = std::chrono::high_resolution_clock::now();
}

void Timer::stop_cpu() {
	_cStop = std::chrono::high_resolution_clock::now();
}

cv::Rect Utils::xywhToxyxy(const cv::Mat& input, const cv::Rect& range)
{
	const float cx = input.at<float>(0);
	const float cy = input.at<float>(1);
	const float cw = input.at<float>(2);
	const float ch = input.at<float>(3);
	cv::Rect box;
	box.x = cvRound(cx - 0.5f * cw);
	box.y = cvRound(cy - 0.5f * ch);
	box.width = cvRound(cw);
	box.height = cvRound(ch);
	return box & range;
}

std::vector<std::string> Utils::getClassName(std::string class_file)
{
	std::ifstream InFile;
	InFile.open(class_file);
	if (!InFile)
	{
		std::cerr << "Failed to open cocoName.txt";
		exit(1);
	}
	std::string classname;
	std::vector<std::string> classnames;
	while (InFile >> classname)
	{
		classnames.push_back(classname);
	}
	InFile.close();
	return classnames;
}

void Utils::PutMask(cv::Mat& image, std::vector<Segmentblob>& results, std::vector<std::string> class_names)
{
	/* 生成随机颜色 */
	std::vector<cv::Scalar> colors;
	std::srand(std::time(nullptr));
	for (int i = 0; i < class_names.size(); i++)
	{
		const int b = std::rand() % 255;
		const int g = std::rand() % 255;
		const int r = std::rand() % 255;
		colors.push_back(cv::Scalar(b, g, r));
	}

	cv::Mat mask = image.clone();
	for (const Segmentblob& result : results)
	{
		cv::rectangle(image, result.box, cv::Scalar(0, 0, 0), 2, 4);
		mask(result.box).setTo(colors[result.class_id], result.mask);
		const std::string label = cv::format("%s:%.2f", class_names[result.class_id].c_str(), result.class_confidence);

		int baseline;
		const cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
		cv::putText(image, label,
			cv::Point(result.box.x, max(result.box.y, float(labelSize.height))),
			cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(100, 100, 200), 3);
	}
	cv::addWeighted(image, 0.5, mask, 0.8, 1, image);
}