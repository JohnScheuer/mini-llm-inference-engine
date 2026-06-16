#ifndef TENSOR_INT8_H
#define TENSOR_INT8_H

#include <vector>
#include <cstdint>

#define QK8_0 32

struct BlockQ8_0 {
    float d;
    int8_t qs[QK8_0];
};

struct Tensor {
    int rows, cols;
    std::vector<float> data;
    std::vector<BlockQ8_0> q_data;

    Tensor() : rows(0), cols(0) {}
    Tensor(int r, int c) : rows(r), cols(c), data(r * c, 0.0f) {}
};

void quantize_row_q8_0(const float* x, BlockQ8_0* y, int k);

#endif