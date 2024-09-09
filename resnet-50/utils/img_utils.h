#ifndef _IMG_UTILS_H
#define _IMG_UTILS_H

#include <opencv2/opencv.hpp>

namespace ImageUtils {

typedef enum {
    RGB = 0,
    BGR,
    GRAY,
} FormatType;

/*******************************************************************************
 * 
 * @brief Convert OpenCV image to vector
 * 
 * @param [in] img OpenCV image
 * @param [in] need_rgb_swap If true, swap BGR to RGB
 * 
 * @return Vector of image pixels
 * 
 *******************************************************************************/
std::vector<float> mat2vector(const cv::Mat& img, std::vector<float> mean_vec,
    std::vector<float> std_vec, bool need_rgb_swap);

}   // namespace ImageUtils

#endif  // _IMG_UTILS_H