#ifndef TRT_UTILS_H
#define TRT_UTILS_H

#include <vector>
#include <fstream>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include "utils/nv_logger.h"

namespace trt_sample {

std::string dims2str(const nvinfer1::Dims &dims);

bool onnx2trtEngine(const std::string& trt_file, const std::string& onnx_file);

nvinfer1::ICudaEngine* loadEngine(const std::string& trt_file, nvinfer1::IRuntime* runtime);

int volume(const nvinfer1::Dims& dims);

int data_type_size(nvinfer1::DataType type);

}   // namespace trt_sample

#endif  // TRT_UTILS_H

