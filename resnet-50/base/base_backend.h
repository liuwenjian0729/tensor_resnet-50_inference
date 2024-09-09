#ifndef _BASE_BACKEND_H
#define _BASE_BACKEND_H

#include <memory>
#include <vector>
#include <unordered_map>
#include "base/base_context.h"

namespace trt_sample {

typedef std::unordered_map<std::string, std::vector<int>> IOShapeMap;

struct BackendInitParams {
    ContextInitParams context_params;
    int max_batch_size;
};

class BaseBackend {
public:
    BaseBackend() = default;
    virtual ~BaseBackend() = default;

    /*****************************************************************************
     * 
     * @brief Initialize the tensorrt backend.
     * 
     * @param [in] params initialization parameters.
     * 
     * @return true if initialization is successful.
     * 
     *****************************************************************************/
    virtual bool Init(const BackendInitParams& params) = 0;

    /*****************************************************************************
     * 
     * @brief do inference.
     * 
     * @param [in] input input image data.
     * @param [out] output output probability.
     * @param [in] stream cuda stream.
     * 
     * @return true if initialization is successful.
     * 
     *****************************************************************************/
    virtual bool Inference(const std::vector<float>& input, std::vector<float>* output, cudaStream_t stream) = 0;

    /*****************************************************************************
     * 
     * @brief get the input/output shapes of the backend.
     * 
     * @return the input/output shapes of the backend.
     * 
     *****************************************************************************/
    virtual IOShapeMap& GetBackendIOShapes() = 0;

protected:
    std::unique_ptr<BaseContext> context_;
};

}   // namespace trt_sample

#endif  // _BASE_BACKEND_H

