#ifndef _TRT_ENGINE_H
#define _TRT_ENGINE_H

#include <vector>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "base/base_backend.h"

namespace trt_sample {

struct EngineParams {
    std::string format;             // image format RBG or BGR
    int32_t width;                  // input image width
    int32_t height;                 // input image height
    std::vector<float> mean_vec;    // mean value for normalization
    std::vector<float> scale_vec;     // std value for normalization
};

class TrtEngine {
public:
    TrtEngine()  = default;
    ~TrtEngine();

    /***********************************************************
     * 
     * @brief parse config file & create a teneorRT backend
     * 
     * @param [in] config_file input config file
     * 
     * @return true if success
     * 
    ************************************************************/
    bool init(const std::string& config_file);

    /***********************************************************
     * 
     * @brief call this method process image for loop
     * 
    ************************************************************/
    void run();

    /***********************************************************
     * 
     * @brief call this method process image for once
     * 
    ************************************************************/
    void run_once();

    /***********************************************************
     * 
     * @brief clear engine
     * 
    ************************************************************/
    void destroy();

private:
    std::unique_ptr<BaseBackend> backend_;
    cudaStream_t stream_;
    EngineParams params_;
};

}   //  namespace trt_sample

#endif  //  _TRT_ENGINE_H

