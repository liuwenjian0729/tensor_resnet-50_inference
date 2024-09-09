#include <cmath>
#include <math.h>
#include "tensorrt/trt_backend.h"
#include "engine/trt_engine.h"
#include "utils/proto_utils.h"
#include "backend_param.pb.h"

namespace trt_sample {

std::vector<float> softmax(const std::vector<float>& input) {
    std::vector<float> probs;

    // max val
    auto get_max = [](const std::vector<float>& vec) -> float {
        float max = std::numeric_limits<float>::lowest();
        for (float val : vec) {
            if (val > max) {
                max = val;
            }
        }
        return max;
    };

    // cal max value
    float max_val = get_max(input);
    // float max_val = 0;
    float sum = 0.0f;

    for (auto val: input) {
        float exp_val = std::exp(val - max_val);
        sum += exp_val;
        probs.emplace_back(exp_val);
    }

    // nomalize
    for (size_t i = 0; i < probs.size(); ++i) {
        probs[i] /= sum;
    }

    return probs;
}

TrtEngine::~TrtEngine() {
    this->destroy();
}

bool TrtEngine::init(const std::string& config_file){
    // 1. load config
    trt_sample::common::BackendParam configs;
    if (!LoadProtoFromTextFile(config_file, &configs)) {
        std::cerr << "Failed to load backend param from text file: " << config_file << std::endl;
        return false;
    }
    // debug
    std::cout << "engine_file: " << configs.engine_file() <<"max batch size: "<<configs.max_batch_size()<< std::endl;

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

void TrtEngine::run() {
    // 1. read image
    cv::Mat image = cv::imread("../scripts/kitten.jpg");

    // 2. preprocess
    cv::resize(image, image, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);
    std::vector<float> input;
    std::vector<float> output;
    input = mat2vector(image);
    std::cout<<"rows: "<<image.rows<<" cols: "<<image.cols<<" channels: "<<image.channels()<<" size: "<<input.size()<<std::endl;

    // 3. inference
    backend_->Inference(input, &output, stream_);

    // 4. post process
    std::vector<float> probs = softmax(output);
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

std::vector<float> TrtEngine::mat2vector(const cv::Mat& img, bool need_rgb_swap) {
    cv::Mat img_float;
    if (need_rgb_swap) {
        cv::cvtColor(img, img_float, cv::COLOR_BGR2RGB);
    } else {
        img.convertTo(img_float, CV_32F);
    }

    int height = img_float.rows;
    int width = img_float.cols;
    int channels = img_float.channels();

    std::vector<float> mean_vec = {0.485, 0.456, 0.406};
    std::vector<float> std_vec = {0.229, 0.224, 0.225};

    // convert HWC to CHW
    std::vector<float> chw_data(channels * height * width);
    int chw_index = 0;

    int index = 0;
    for (int c = 0; c < channels; ++c) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                auto pixel = img.ptr<cv::Vec3b>(i);
                float value = pixel[j][c];
                chw_data[chw_index++] = ((value / 255) - mean_vec[c]) / std_vec[c];
            }
        }
    }

    return chw_data;
}

void TrtEngine::destroy() {
    cudaStreamDestroy(stream_);
    backend_.reset();
}

} // namespace name


