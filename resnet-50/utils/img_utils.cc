#include "utils/img_utils.h"

namespace ImageUtils {

///////////////////////////////////////////////////////////////////
// mat2vector()
///////////////////////////////////////////////////////////////////
std::vector<float> mat2vector(const cv::Mat& img, const std::vector<float>& mean_vec,
    const std::vector<float>& std_vec, bool need_rgb_swap) {
    cv::Mat img_float;
    if (need_rgb_swap) {
        cv::cvtColor(img, img_float, cv::COLOR_BGR2RGB);
    } else {
        img.convertTo(img_float, CV_32F);
    }

    int height = img_float.rows;
    int width = img_float.cols;
    int channels = img_float.channels();

    // convert HWC to CHW
    std::vector<float> chw_data(channels * height * width);
    int chw_index = 0;

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

}   // namespace ImageUtils
