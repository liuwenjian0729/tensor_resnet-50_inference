#include <iostream>
#include "tensorrt/trt_backend.h"
#include "tensorrt/trt_context.h"

namespace trt_sample {

bool TrtBackend::Init(const BackendInitParams& params) {
    context_.reset(new TrtContext());
    if(!context_->init(params.context_params)) {
        std::cerr << "TrtContext init failed" << std::endl;
        return false;
    }

    return true;
}

bool TrtBackend::Inference() {

        return true;
}

}   // namespace trt_sample


