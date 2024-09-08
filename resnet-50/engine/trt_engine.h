#ifndef _TRT_ENGINE_H
#define _TRT_ENGINE_H

#include <vector>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "base/base_backend.h"

namespace trt_sample {

class TrtEngine {
public:
    TrtEngine()  = default;
    ~TrtEngine();

    bool init(const std::string& config_file);
    void run();
    void run_once();
    void destroy();

private:
    std::vector<float> mat2vector(const cv::Mat& img, bool need_rgb_swap = true);

    std::unique_ptr<BaseBackend> backend_;
    cudaStream_t stream_;
};

}   //  namespace trt_sample

#endif  //  _TRT_ENGINE_H

