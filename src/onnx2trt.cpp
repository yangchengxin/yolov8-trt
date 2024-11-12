#include "onnx2trt.h"
#include <NvInferPlugin.h>
#define _CRT_SECURE_NO_WARNINGS
#define FP16 false

int onnx_to_engin(const char* onnx_model, const char* engine_model)
{
	//1.构建builder，并使用builder构建Network用于存储模型信息
	//2.使用Network构建parser用于从onnx文件中解析模型信息并回传给Network
	//3.使用builder构建profile用于设置动态维度，并从dynamicBinding中获取动态维度信息
	//4.构建calibrator用于校准模型，并通过BatchStream加载校准数据集
	//5.使用Builder构建Config用于设置生成Engine的参数，包括Calibrator和Profile
	//6.Builder使用Network中的模型信息和Config中的参数来生成Engine以及校准参数calParameter
	//7.通过BatchStream加载待测试数据集并传入engine，最终输出结果

	// Create builder
	TRTLogger gLogger;
	auto builder = createInferBuilder(gLogger);

	const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	IBuilderConfig* config = builder->createBuilderConfig();

	// Create model to populate the network
	INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

	// Parse ONNX file
	auto parser = nvonnxparser::createParser(*network, gLogger);

	bool parser_status = parser->parseFromFile(onnx_model, 3);

	// Get the name of network input
	Dims dim = network->getInput(0)->getDimensions();

	if (dim.d[0] == -1)  // -1 means it is a dynamic model
	{
		const char* name = network->getInput(0)->getName();
		IOptimizationProfile* profile = builder->createOptimizationProfile();
		profile->setDimensions(name, OptProfileSelector::kMIN, Dims4(1, dim.d[1], dim.d[2], dim.d[3]));
		profile->setDimensions(name, OptProfileSelector::kOPT, Dims4(4, dim.d[1], dim.d[2], dim.d[3]));
		profile->setDimensions(name, OptProfileSelector::kMAX, Dims4(20, dim.d[1], dim.d[2], dim.d[3]));
		config->addOptimizationProfile(profile);
		//config->setCalibrationProfile()
	}

	// Build engine
	config->setMaxWorkspaceSize(1 << 30);
	if (FP16) {
		config->setFlag(nvinfer1::BuilderFlag::kFP16); // 设置精度计算
		//config->setFlag(nvinfer1::BuilderFlag::kINT8);
	}

	ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	//builder->buildSerializedNetwork()
	// Serialize the model to engine file
	IHostMemory* modelStream{ nullptr };
	assert(engine != nullptr);
	modelStream = engine->serialize();

	std::ofstream p(engine_model, std::ios::binary);
	if (!p) {
		std::cerr << "could not open output file to save model" << std::endl;
		return -1;
	}
	p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
	std::cout << "generate file success!" << std::endl;

	// Release resources
	modelStream->destroy();
	network->destroy();
	engine->destroy();
	//builder->destroy();
	config->destroy();
	return 0;
}

bool build_model() 
{
	
	//1.构建builder，并使用builder构建Network用于存储模型信息
	//2.使用Network构建parser用于从onnx文件中解析模型信息并回传给Network
	//3.使用builder构建profile用于设置动态维度，并从dynamicBinding中获取动态维度信息
	//4.构建calibrator用于校准模型，并通过BatchStream加载校准数据集
	//5.使用Builder构建Config用于设置生成Engine的参数，包括Calibrator和Profile
	//6.Builder使用Network中的模型信息和Config中的参数来生成Engine以及校准参数calParameter
	//7.通过BatchStream加载待测试数据集并传入engine，最终输出结果
	
	TRTLogger logger;
	
	// 这是基本需要的组件
	/*auto builder = make_nvshared(nvinfer1::createInferBuilder(logger));
	auto config = make_nvshared(builder->createBuilderConfig());
	auto network = make_nvshared(builder->createNetworkV2(1));*/
	
	auto builder = nvinfer1::createInferBuilder(logger);
	auto config = builder->createBuilderConfig();
	auto network = builder->createNetworkV2(1);
	
	// 通过onnxparser解析器解析的结果会填充到network中，类似addConv的方式添加进去
	auto parser = make_nvshared(nvonnxparser::createParser(*network, logger));
	if (!parser->parseFromFile("best.onnx", 1)) {
		printf("Failed to parse onnx\n");
	
		// 注意这里的几个指针还没有释放，是有内存泄漏的，后面考虑更优雅的解决
		return false;
	}
	
	int maxBatchSize = 1;
	printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);
	config->setMaxWorkspaceSize(1 << 28);
	
	
	// 如果模型有多个输入，则必须多个profile
	auto profile = builder->createOptimizationProfile();
	auto input_tensor = network->getInput(0);
	auto input_dims = input_tensor->getDimensions();
	
	// 配置最小、最优、最大范围
	input_dims.d[0] = 1;
	profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
	profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
	input_dims.d[0] = maxBatchSize;
	profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
	config->addOptimizationProfile(profile);
	config->setHardwareCompatibilityLevel(nvinfer1::HardwareCompatibilityLevel::kNONE);
	//设置成硬件兼容模式
	
	auto engine = make_nvshared(builder->buildEngineWithConfig(*network, *config));
	if (engine == nullptr) {
		printf("Build engine failed.\n");
		return false;
	}
	
	// 将模型序列化，并储存为文件
	auto model_data = make_nvshared(engine->serialize());
	FILE* f;
	errno_t err	= fopen_s(&f, "best.trtmodel", "wb");
	fwrite(model_data->data(), 1, model_data->size(), f);
	fclose(f);
	
	// 卸载顺序按照构建顺序倒序
	printf("Build Done.\n");
	return true;
}

//int main(int argc, char** argv)
//{
//	const char* onnx_path   = "best.onnx";
//	const char* engine_path = "best.engine";
//	onnx_to_engin(onnx_path, engine_path);
//
//	return 0;
//	system("pause");
//}