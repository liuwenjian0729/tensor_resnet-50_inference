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
    float maxp =  INT_MIN;
    int index = 0;
    for (int i=0; i< output.size(); ++i) {
	if(output[i]>maxp) {
	    maxp = output[i];
	    index = i;
	}
    }
    std::cout<<"maxp: "<<maxp<<" index: "<<index<<std::endl;
}

std::vector<float> TrtEngine::mat2vector(const cv::Mat& img, bool need_rgb_swap) {
    cv::Mat img_float;
    if (need_rgb_swap) {
        cv::cvtColor(img, img_float, cv::COLOR_BGR2RGB);
    } else {
        img.convertTo(img_float, CV_32F);
    }

    // normalize
    std::vector<float> mean_vec = {0.485, 0.456, 0.406};
    std::vector<float> std_vec = {0.229, 0.224, 0.225};

    std::vector<float> norm_image;
    norm_image.resize(img_float.rows * img_float.cols * 3);

    int count = 0;
    for(int i = 0; i<img_float.rows; i++){
        uchar* uc_pixel = img_float.data + i * img_float.step;
        for(int j = 0; j<img_float.cols; j++) {
            norm_image[count] = (uc_pixel[0] / 255. - mean_vec[0]) / std_vec[0];
            norm_image[count + img_float.rows * img_float.cols] = (uc_pixel[1] / 255. - mean_vec[1]) / std_vec[1];
            norm_image[count + 2 * img_float.rows * img_float.cols] = (uc_pixel[2] / 255. - mean_vec[2]) / std_vec[2];
            uc_pixel += 3;
            count++;
        }
    } 

    return norm_image;
}

void TrtEngine::destroy() {
    cudaStreamDestroy(stream_);
    backend_.reset();
}

} // namespace name


