#include "tensorrt/trt_context.h"
#include "utils/nv_utils.h"
#include "utils/nv_logger.h"

namespace trt_sample {

bool TrtContext::init(const ContextInitParams& params) {
    runtime_.reset(nvinfer1::createInferRuntime(gLogger));
   
    // if (!onnx2trtEngine(params.trt_file, params.onnx_file)) {
    //    std::cerr << "Failed to convert onnx" << std::endl;
    //	return false;
    // }

    engine_.reset(loadEngine(params.trt_file, runtime_.get()));
    if (engine_.get() == nullptr) {
        std::cerr << "Failed to load engine" << std::endl;
        return false;
    }

    context_.reset(engine_->createExecutionContext());
    if (context_.get() == nullptr) {
        std::cerr << "Failed to create context" << std::endl;
        return false;
    }
	
    return true;
}

void TrtContext::info() {
        int profile_idx = context_->getOptimizationProfile();
        const nvinfer1::ICudaEngine &engine = context_->getEngine();

        for (int i = 0; i < engine.getNbBindings() / engine.getNbOptimizationProfiles(); i++) {
                if (engine.bindingIsInput(i)) {
                        std::string input_name(engine.getBindingName(i));
                        int binding_index = profile_idx * engine.getNbBindings() / engine.getNbOptimizationProfiles() +
                          engine.getBindingIndex(input_name.c_str());
                        nvinfer1::Dims dims = context_->getBindingDimensions(binding_index);
                        nvinfer1::TensorFormat format = engine.getBindingFormat(binding_index);
                        nvinfer1::DataType type = engine.getBindingDataType(binding_index);

                        context_->setBindingDimensions(binding_index, dims);
                        std::cout<<"input name: "<<input_name<<"\n"     \
                                <<"binding idx: "<<binding_index<<"\n"  \
                                <<"shape: "<<dims2str(dims)<<std::endl;
                } else {
                        std::string output_name(engine.getBindingName(i));
                        int binding_index = profile_idx * engine.getNbBindings() / engine.getNbOptimizationProfiles() +
                          engine.getBindingIndex(output_name.c_str());

                        nvinfer1::Dims dims = context_->getBindingDimensions(binding_index);
                        nvinfer1::TensorFormat format = engine.getBindingFormat(binding_index);
                        nvinfer1::DataType type = engine.getBindingDataType(binding_index);

                        std::cout<<"input name: "<<output_name<<"\n"     \
                                <<"binding idx: "<<binding_index<<"\n"  \
                                <<"shape: "<<dims2str(dims)<<std::endl;
                }
        }
}

}   // namespace trt_sample

