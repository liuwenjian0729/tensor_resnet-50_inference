#ifndef _BASE_CONTEXT_H
#define _BASE_CONTEXT_H

#include <iostream>
#include <NvInfer.h>
#include <string>

namespace trt_sample {

struct ContextInitParams {
    std::string trt_file;
    std::string onnx_file;
};

class BaseContext {
public:
    BaseContext() = default;
    virtual ~BaseContext() = default;

    /*****************************************************************************
     * 
     * @brief init tensotrrt context
     * 
     * @param [in] init_params params for init context
     * 
     * @return true init success
     * 
     *****************************************************************************/
    virtual bool init(const ContextInitParams &init_params) = 0;

    /*****************************************************************************
     * 
     * @brief get IExecutionContext pointer object
     * 
     * @return pointer of IExecutionContext
     * 
     *****************************************************************************/
    nvinfer1::IExecutionContext* get_context() { return context_.get(); }

protected:
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
};

}   // namespace trt_sample

#endif  // _BASE_CONTEXT_H


