#ifndef _NET_UTILS_H
#define _NET_UTILS_H

#include <stdint.h>
#include <string>
#include <vector>

namespace NetworkUtils {

/*******************************************************************************
 * 
 * @brief Softmax function
 * 
 * @param [in] vals vector of output values
 * 
 * @return vector of probabilities
 * 
 *******************************************************************************/
std::vector<float> Softmax(const std::vector<float> &vals);

}   // namespace NetworkUtils

#endif  // _NET_UTILS_H
