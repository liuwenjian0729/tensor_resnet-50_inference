#ifndef _BASE_BACKEND_H
#define _BASE_BACKEND_H

#include <memory>
#include "base/base_context.h"

namespace trt_sample {

struct BackendInitParams {
    ContextInitParams context_params;
};

class BaseBackend {
public:
    BaseBackend() = default;
    virtual ~BaseBackend() = default;

    virtual bool Init(const BackendInitParams& params) = 0;
    virtual bool Inference() = 0;

protected:
    std::unique_ptr<BaseContext> context_;
};

}   // namespace trt_sample

#endif  // _BASE_BACKEND_H
