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
        ~TrtBackend();

        bool Init(const BackendInitParams& params) override;
        bool Inference(const std::vector<float>& input, std::vector<float>* output, cudaStream_t stream) override;
        void destroy();
        IOShapeMap& GetBackendIOShapes() override;

    private:
        void binding_io_buffer(int index, void* buffer);
        std::vector<void*> buffers_;
        std::unordered_map<int, std::pair<std::string, int>> io_shapes_;
        int max_batch_size_;
};

}   // namespace trt_sample

#endif  // _TRT_BACKEND_H

