#include "tensorrt/trt_backend.h"
#include "engine/trt_engine.h"
#include "utils/proto_utils.h"
#include "backend_param.pb.h"

namespace trt_sample {

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
    cv::Mat image = cv::imread("/data/workspace/CODE/test/cifar-10/cat.jpeg");

    // 2. preprocess
    cv::resize(image, image, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);
    std::vector<float> input;
    std::vector<float> output;
    input = mat2vector(image);
    std::cout<<"rows: "<<image.rows<<" cols: "<<image.cols<<" channels: "<<image.channels()<<" size: "<<input.size()<<std::endl;

    // 3. inference
    backend_->Inference(input, &output, stream_);

    // 4. post process
    std::cout<<"size of prob: "<<output.size()<<std::endl;
}

std::vector<float> TrtEngine::mat2vector(const cv::Mat& img, bool need_rgb_swap) {
    cv::Mat img_float;
    if (need_rgb_swap) {
        cv::cvtColor(img, img_float, cv::COLOR_BGR2RGB);
    } else {
        img.convertTo(img_float, CV_32F);
    }

    // normalize
    img_float /= 255.0f;

    // read to vector
    int rows = img_float.rows, cols = img_float.cols, channels = img_float.channels();
    std::vector<float> img_vec(rows * cols * channels);
   for (int y = 0; y < rows; y++) {
       for (int x = 0; x < cols; x++) {
           for (int c = 0; c < channels; c++) {
               img_vec[(y * cols + x) * channels + c] = img_float.at<float>(y, x * channels + c);
           }
       }
   }
    return img_vec;
}

void TrtEngine::destroy() {
    cudaStreamDestroy(stream_);
    backend_.reset();
}

} // namespace name

