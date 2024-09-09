/*
 * @Author: liuwenjian1523@163.com
 * @Date: 2024-09-07 23:12:45
 * @LastEditors: liuwenjian1523@163.com
 * @LastEditTime: 2024-09-09 15:23:32
 * @Description: 请填写简介
 */
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
    std::vector<float> mat2vector(const cv::Mat& img, bool need_rgb_swap = true);

    std::unique_ptr<BaseBackend> backend_;
    cudaStream_t stream_;
};

}   //  namespace trt_sample

#endif  //  _TRT_ENGINE_H

