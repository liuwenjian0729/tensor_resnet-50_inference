#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <sstream>
#include "utils/nv_utils.h"
#include "tensorrt/trt_context.h"

using namespace trt_sample;

int main() {
    ContextInitOptions options;
    options.trt_file = "/data/workspace/CODE/TensorRT_inference/models/resnet50.engine";

    std::unique_ptr<TrtContext> context_ptr(new TrtContext());

    if (!context_ptr->init(options)) {
    	std::cerr<<"Failed to create TrtContext object"<<std::endl;
	return -1;
    }

    std::cout<<std::endl;
    context_ptr->info();

    std::cout<< "----Done----"<<std::endl;

    return 0;
}
