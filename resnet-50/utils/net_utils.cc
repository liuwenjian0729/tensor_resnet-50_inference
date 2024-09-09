#include <cmath>
#include <math.h>
#include "utils/net_utils.h"

namespace NetworkUtils {

///////////////////////////////////////////////////////////////////
// Softmax()
///////////////////////////////////////////////////////////////////
std::vector<float> Softmax(const std::vector<float> &vals) {
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
    float max_val = get_max(vals);
    // float max_val = 0;
    float sum = 0.0f;

    for (auto val: vals) {
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

}   // namespace NetworkUtils