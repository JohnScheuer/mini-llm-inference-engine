#ifndef KERNEL_AVX2_H
#define KERNEL_AVX2_H

#include <immintrin.h>

// Micro-kernel 6x16 acumulativo (C += A * B)
static inline void micro_kernel_6x16_acc(int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
    // Carrega o que já existe em C para os registradores
    __m256 c00 = _mm256_loadu_ps(&C[0 * ldc + 0]); __m256 c01 = _mm256_loadu_ps(&C[0 * ldc + 8]);
    __m256 c10 = _mm256_loadu_ps(&C[1 * ldc + 0]); __m256 c11 = _mm256_loadu_ps(&C[1 * ldc + 8]);
    __m256 c20 = _mm256_loadu_ps(&C[2 * ldc + 0]); __m256 c21 = _mm256_loadu_ps(&C[2 * ldc + 8]);
    __m256 c30 = _mm256_loadu_ps(&C[3 * ldc + 0]); __m256 c31 = _mm256_loadu_ps(&C[3 * ldc + 8]);
    __m256 c40 = _mm256_loadu_ps(&C[4 * ldc + 0]); __m256 c41 = _mm256_loadu_ps(&C[4 * ldc + 8]);
    __m256 c50 = _mm256_loadu_ps(&C[5 * ldc + 0]); __m256 c51 = _mm256_loadu_ps(&C[5 * ldc + 8]);

    for (int k = 0; k < K; ++k) {
        __m256 b0 = _mm256_loadu_ps(&B[k * ldb + 0]);
        __m256 b1 = _mm256_loadu_ps(&B[k * ldb + 8]);
        __m256 a;

        a = _mm256_set1_ps(A[0 * lda + k]);
        c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
        a = _mm256_set1_ps(A[1 * lda + k]);
        c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
        a = _mm256_set1_ps(A[2 * lda + k]);
        c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
        a = _mm256_set1_ps(A[3 * lda + k]);
        c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);
        a = _mm256_set1_ps(A[4 * lda + k]);
        c40 = _mm256_fmadd_ps(a, b0, c40); c41 = _mm256_fmadd_ps(a, b1, c41);
        a = _mm256_set1_ps(A[5 * lda + k]);
        c50 = _mm256_fmadd_ps(a, b0, c50); c51 = _mm256_fmadd_ps(a, b1, c51);
    }

    _mm256_storeu_ps(&C[0 * ldc + 0], c00); _mm256_storeu_ps(&C[0 * ldc + 8], c01);
    _mm256_storeu_ps(&C[1 * ldc + 0], c10); _mm256_storeu_ps(&C[1 * ldc + 8], c11);
    _mm256_storeu_ps(&C[2 * ldc + 0], c20); _mm256_storeu_ps(&C[2 * ldc + 8], c21);
    _mm256_storeu_ps(&C[3 * ldc + 0], c30); _mm256_storeu_ps(&C[3 * ldc + 8], c31);
    _mm256_storeu_ps(&C[4 * ldc + 0], c40); _mm256_storeu_ps(&C[4 * ldc + 8], c41);
    _mm256_storeu_ps(&C[5 * ldc + 0], c50); _mm256_storeu_ps(&C[5 * ldc + 8], c51);
}

#endif