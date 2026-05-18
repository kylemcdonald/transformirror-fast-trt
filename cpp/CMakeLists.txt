cmake_minimum_required(VERSION 3.24)
project(sdxl_turbo_trt_runner LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(trt_sdxl_runner trt_sdxl_runner.cu)
target_include_directories(
    trt_sdxl_runner
    PRIVATE
    /usr/include/x86_64-linux-gnu
    /usr/local/cuda/include
)
target_link_libraries(
    trt_sdxl_runner
    PRIVATE
    nvinfer
    nvinfer_plugin
    cudart
)

add_executable(transformirror_fast_app transformirror_fast_app.cu)
target_include_directories(
    transformirror_fast_app
    PRIVATE
    /usr/include/x86_64-linux-gnu
    /usr/local/cuda/include
)
target_link_libraries(
    transformirror_fast_app
    PRIVATE
    nvinfer
    nvinfer_plugin
    cudart
    jpeg
    GLEW
    GL
    X11
    pthread
)
