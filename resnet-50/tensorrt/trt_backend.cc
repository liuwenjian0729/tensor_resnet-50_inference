#include <iostream>
#include "tensorrt/trt_backend.h"
#include "tensorrt/trt_context.h"
#include "utils/nv_utils.h"

namespace trt_sample {

TrtBackend::~TrtBackend() {
    this->destroy();
}

bool TrtBackend::Init(const BackendInitParams& params) {
    // 1. create context
    context_.reset(new TrtContext());
    if(!context_->init(params.context_params)) {
        std::cerr << "TrtContext init failed" << std::endl;
        return false;
    }
    max_batch_size_ = params.max_batch_size;

    // 2. malloc gpu memory
    nvinfer1::IExecutionContext *context = context_->get_context();
    if (nullptr == context) {
    	std::cerr<<"context is null"<<std::endl;
	return false;
    }
    const nvinfer1::ICudaEngine &engine = context->getEngine();
    buffers_.resize(10);
    for (int i = 0; i < engine.getNbBindings(); ++i) {
        std::string name(engine.getBindingName(i));
        nvinfer1::Dims dims = engine.getBindingDimensions(i);
        // dynamic shape
        if (dims.d[0] == -1) {
            dims.d[0] = max_batch_size_;
        }
        nvinfer1::DataType dtype = engine.getBindingDataType(i);
        int64_t totalSize = trt_sample::volume(dims) * trt_sample::data_type_size(dtype);
        cudaMalloc(&buffers_[i], totalSize);
        io_shapes_[i] = {name, totalSize};
    }
    return true;
}

bool TrtBackend::Inference(const std::vector<float>& input, std::vector<float>* output, cudaStream_t stream) {
    nvinfer1::IExecutionContext *context = nullptr;
    context = context_->get_context();
    const nvinfer1::ICudaEngine &engine = context->getEngine();

    // copy input data to gpu
    for (const auto& iter : io_shapes_) {
        int index = iter.first;
        auto param = iter.second;
        if(engine.bindingIsInput(index)) {
            std::cout<<"input name:"<<param.first<<" input size:"<<param.second<<std::endl;
            cudaMemcpyAsync(buffers_[index], input.data(), param.second, cudaMemcpyHostToDevice, stream);
            break;
        }
    }

    for (auto it: input) {
        //std::cout<<it<<" ";
    }

    // inference
    bool succ = true;
    succ = context->enqueueV2(buffers_.data(), stream, nullptr);
    if (!succ) {
    	std::cerr<<"Failed to infer"<<std::endl;
	return false;
    }

    // copy output data to cpu
    for (const auto& iter : io_shapes_) {
        int index = iter.first;
        auto param = iter.second;
        if(!engine.bindingIsInput(index)) {
	        output->resize(1000);
            cudaMemcpyAsync(output->data(), buffers_[index], param.second, cudaMemcpyDeviceToHost, stream);
            break;
        }
    }

    return true;
}

void TrtBackend::destroy() {
    for (int i = 0; i < buffers_.size(); i++) {
        cudaFree(buffers_[i]);
    }
}

IOShapeMap& TrtBackend::GetBackendIOShapes() {

}

}   // namespace trt_sample

