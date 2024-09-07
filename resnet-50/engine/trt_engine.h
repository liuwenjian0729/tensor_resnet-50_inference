#ifndef _TRT_ENGINE_H
#define _TRT_ENGINE_H
#include "base/base_backend.h"

namespace trt_sample {

class TrtEngine {
public:
    TrtEngine() = default;
    ~TrtEngine() = default;

    bool init(const std::string& config_file);
    void run();
    void destroy();

private:
    std::unique_ptr<BaseBackend> backend_;
};

}   //  namespace trt_sample

#endif  //  _TRT_ENGINE_H
