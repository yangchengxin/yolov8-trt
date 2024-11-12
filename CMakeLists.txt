cmake_minimum_required(VERSION 3.20)
project (yolov8-seg)
add_definitions(-std=c++14)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
MESSAGE(STATUS "current platform: Windows") 
set(CUDA_COMPILER_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7/bin/nvcc.exe") 
set(TENSORRT_PATH "C:/TensorRT-8.6.1.6") 

# CUDA
include_directories(${PROJECT_SOURCE_DIR}/windows) 
find_package(CUDA REQUIRED) 
include_directories(${CUDA_INCLUDE_DIRS}) 
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)

# tensorrt
include_directories(${TENSORRT_PATH}/include)
link_directories(${TENSORRT_PATH}/lib)

# OpenCV
find_package(OpenCV REQUIRED
            CONFIG
            PATHS C:/opencv/build/x64/vc16/lib)
include_directories(${OpenCV_INCLUDE_DIRS})
if (OpenCV_FOUND) 
    include_directories(${OpenCV_INCLUDE_DIRS}) 
    link_directories(${OpenCV_LIBRARY_DIRS}) 
    add_definitions(${OpenCV_DEFINITIONS}) 
else() 
    message(FATAL_ERROR "OpenCV not found!") 
endif()

include_directories(${PROJECT_SOURCE_DIR}/include/)
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp)
add_executable (yolov8-seg ${PROJECT_SOURCE_DIR}/src/main.cpp ${SRCS})

# 添加动态链接库
target_link_libraries(yolov8-seg nvinfer cudart nvonnxparser cudnn nvinfer_plugin ${OpenCV_LIBS})
# 定义预处理器头
target_compile_definitions(yolov8-seg PRIVATE INFERENCE)