#include "matmul_blocked.h"
#include <immintrin.h>
#include <cstring>
#include <omp.h>

void matmul_blocked(int M, int N, int K, const float* A, const float* B, float* C) {
    if (M == 1) {
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < N; j++) {
            float sum = 0;
            const float* w = &B[j * K];
            for (int k = 0; k < K; k++) {
                sum += A[k] * w[k];
            }
            C[j] = sum;
        }
        return;
    }
    // Fallback naive para M > 1
    std::memset(C, 0, (size_t)M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a = A[i * K + k];
            for (int j = 0; j < N; j++) C[i * N + j] += a * B[k * N + j];
        }
    }
}