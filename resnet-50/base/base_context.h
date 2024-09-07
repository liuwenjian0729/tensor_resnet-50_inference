#ifndef _BASE_CONTEXT_H
#define _BASE_CONTEXT_H

#include <string>

namespace trt_sample {

struct ContextInitParams {
    std::string trt_file;
};

class BaseContext {
public:
    BaseContext() = default;
    virtual ~BaseContext() = default;

    virtual bool init(const ContextInitParams &init_params) = 0;
};

}   // namespace trt_sample

#endif  // _BASE_CONTEXT_H

