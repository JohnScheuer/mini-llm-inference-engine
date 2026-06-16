#ifndef MATMUL_BLOCKED_H
#define MATMUL_BLOCKED_H

#include "tensor_int8.h"

void matmul_gemv_int8_avx2(int N, int K,
                           const BlockQ8_0* W,
                           const BlockQ8_0* X, float* Y);

void matmul_blocked_int8(int M, int N, int K,
                         const BlockQ8_0* W,
                         const float* X_fp32, float* Y,
                         BlockQ8_0* X_q_buf);

#endif