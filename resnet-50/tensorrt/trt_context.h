#ifndef _TRT_CONTEXT_H
#define _TRT_CONTEXT_H

#include <memory>
#include <NvInfer.h>
#include "base/base_context.h"

namespace trt_sample {

class TrtContext:public BaseContext {
        public:
            TrtContext() = default;
            ~TrtContext() = default;

            bool init(const ContextInitParams& params) override;

            nvinfer1::IExecutionContext* get_context() { return context_.get(); }

	    void info();
};

}   // namespace trt_sample

#endif  // _TRT_CONTEXT_H

