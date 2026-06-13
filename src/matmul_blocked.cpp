#include "matmul_blocked.h"
#include <immintrin.h>
#include <cstring>
#include <omp.h>

void matmul_blocked(int M, int N, int K, const float* A, const float* B, float* C) {
    if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0) return;

    if (M == 1) {
        // Camada final (Logits) é grande, usamos threads. Camadas de Attention são pequenas, rodamos serial.
        int threads = (N > 4096) ? omp_get_max_threads() : 1;
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (int j = 0; j <= N - 8; j += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                acc = _mm256_fmadd_ps(_mm256_set1_ps(A[k]), _mm256_loadu_ps(&B[k * N + j]), acc);
            }
            _mm256_storeu_ps(&C[j], acc);
        }
        for (int j = (N / 8) * 8; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++) sum += A[k] * B[k * N + j];
            C[j] = sum;
        }
        return;
    }

    std::memset(C, 0, (size_t)M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = A[i * K + k];
            for (int j = 0; j < N; j++) C[i * N + j] += a_ik * B[k * N + j];
        }
    }
}