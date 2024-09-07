#include "tensorrt/trt_backend.h"
#include "engine/trt_engine.h"
#include "utils/proto_utils.h"
#include "backend_param.pb.h"

namespace trt_sample {

bool TrtEngine::init(const std::string& config_file){
    // 1. load config
    trt_sample::common::BackendParam configs;
    if (!LoadProtoFromTextFile(config_file, &configs)) {
        std::cerr << "Failed to load backend param from text file: " << config_file << std::endl;
        return false;
    }
    // debug
    std::cout << "engine_file: " << configs.engine_file() << std::endl;

    // 2. create backend
    backend_.reset(new TrtBackend());

    // 3. config backend
    BackendInitParams params;
    params.context_params.trt_file = configs.engine_file();
    if(!backend_->Init(params)) {
        std::cerr << "Failed to init backend" << std::endl;
        return false;
    }

    return true;
}

void TrtEngine::run() {
    std::cout<<"dummy_run..."<<std::endl;
}

void TrtEngine::destroy() {
    std::cout<<"dummy_destroy..."<<std::endl;
}
    
} // namespace name

