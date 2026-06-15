#include "tensor_int8.h"
#include <cmath>
#include <algorithm>

void quantize_row_q8_0(const float* x, BlockQ8_0* y, int k) {
    int num_blocks = k / QK8_0;
    for (int i = 0; i < num_blocks; ++i) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; ++j) {
            amax = std::max(amax, std::abs(x[i * QK8_0 + j]));
        }
        float d = amax / 127.0f;
        y[i].d = d;
        float id = (d != 0.0f) ? 1.0f / d : 0.0f;
        for (int j = 0; j < QK8_0; ++j) {
            y[i].qs[j] = (int8_t)std::round(x[i * QK8_0 + j] * id);
        }
    }
}