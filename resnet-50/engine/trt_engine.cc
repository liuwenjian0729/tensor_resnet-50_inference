#include "tensorrt/trt_backend.h"
#include "engine/trt_engine.h"
#include "utils/proto_utils.h"
#include "utils/img_utils.h"
#include "utils/net_utils.h"
#include "engine_param.pb.h"

namespace trt_sample {

///////////////////////////////////////////////////////////////////
// TrtEngine::~TrtEngine()
///////////////////////////////////////////////////////////////////
TrtEngine::~TrtEngine() {
    this->destroy();
}

///////////////////////////////////////////////////////////////////
// TrtEngine::init()
///////////////////////////////////////////////////////////////////
bool TrtEngine::init(const std::string& config_file){
    // 1. load config
    trt_sample::common::BackendParam configs;
    if (!LoadProtoFromTextFile(config_file, &configs)) {
        std::cerr << "Failed to load backend param from text file: " << config_file << std::endl;
        return false;
    }

    params_.format = configs.format();
    params_.width = configs.width();
    params_.height = configs.height();
    for (auto& val : configs.mean()) {
        params_.mean_vec.emplace_back(val);
    }
    for (auto& val : configs.scale()) {
        params_.scale_vec.emplace_back(val);
    }

    // 2. create backend
    backend_.reset(new TrtBackend());

    // 3. config backend
    BackendInitParams params;
    params.context_params.trt_file = configs.engine_file();
    params.max_batch_size = configs.max_batch_size();
    if(!backend_->Init(params)) {
        std::cerr << "Failed to init backend" << std::endl;
        return false;
    }

    // 4. create stream
    cudaError_t err = cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, 0);
    if (err != cudaSuccess) {
        std::cerr << "Error creating stream: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    return true;
}

///////////////////////////////////////////////////////////////////
// TrtEngine::run()
///////////////////////////////////////////////////////////////////
void TrtEngine::run() {
    // 1. read image
    cv::Mat image = cv::imread("../scripts/kitten.jpg");

    // 2. preprocess
    cv::resize(image, image, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);
    std::vector<float> input;
    std::vector<float> output;

    input = ImageUtils::mat2vector(image, params_.mean_vec, params_.scale_vec);

    // 3. inference
    backend_->Inference(input, &output, stream_);

    // 4. post process
    std::vector<float> probs = NetworkUtils::Softmax(output);
    float prob = std::numeric_limits<float>::lowest();
    int index = 0;
    for (int i=0; i< probs.size(); ++i) {
        if(probs[i] > prob) {
            prob = probs[i];
            index = i;
        }
    }
    std::cout<<"prob: "<<prob<<" index: "<<index<<std::endl;
}

///////////////////////////////////////////////////////////////////
// TrtEngine::destroy()
///////////////////////////////////////////////////////////////////
void TrtEngine::destroy() {
    cudaStreamDestroy(stream_);
    backend_.reset();
}

} // namespace name


