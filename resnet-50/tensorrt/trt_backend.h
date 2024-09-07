#ifndef _TRT_BACKEND_H
#define _TRT_BACKEND_H

#include <vector>
#include <NvInfer.h>
#include <unordered_map>
#include "base/base_backend.h"

namespace trt_sample {

class TrtBackend: public BaseBackend {
    public:
        TrtBackend() = default;
        ~TrtBackend() = default;

        bool Init(const BackendInitParams& params) override;
        bool Inference() override;

    private:
        std::vector<void*> buffers_;
};

}   // namespace trt_sample

#endif  // _TRT_BACKEND_H
