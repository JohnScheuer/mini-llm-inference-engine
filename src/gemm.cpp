#include "gemm.h"
#include "matmul_blocked.h"

void gemm(
    int M, int N, int K,
    const float* A,
    const float* B,
    float* C
) {
    matmul_blocked(M, N, K, A, B, C);
}