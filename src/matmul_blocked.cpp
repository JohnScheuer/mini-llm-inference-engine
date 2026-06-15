#include "matmul_blocked.h"
#include <immintrin.h>
#include <vector>
#include <omp.h>

void matmul_gemv_int8_avx2(int N, int K, const BlockQ8_0* W, const BlockQ8_0* X, float* Y) {
    const int num_blocks_k = K / QK8_0;
    const __m256i ones_16 = _mm256_set1_epi16(1);

    // Threads ligadas apenas para matrizes grandes (N > 512) para evitar overhead de sincronização
    #pragma omp parallel for if(N > 512) schedule(static)
    for (int i = 0; i < N; ++i) {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        const BlockQ8_0* W_row = W + (i * num_blocks_k);

        int b = 0;
        // 2x Unroll: Processa 64 elementos por ciclo de CPU
        for (; b <= num_blocks_k - 2; b += 2) {
            __m256i w0 = _mm256_loadu_si256((const __m256i*)W_row[b].qs);
            __m256i x0 = _mm256_loadu_si256((const __m256i*)X[b].qs);
            __m256i w1 = _mm256_loadu_si256((const __m256i*)W_row[b+1].qs);
            __m256i x1 = _mm256_loadu_si256((const __m256i*)X[b+1].qs);

            // maddubs chain (u8*s8 -> i16 -> i32) com truque de sinal
            __m256i d0 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(x0), _mm256_sign_epi8(w0, x0)), ones_16);
            __m256i d1 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(x1), _mm256_sign_epi8(w1, x1)), ones_16);

            acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d0), _mm256_set1_ps(W_row[b].d * X[b].d), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d1), _mm256_set1_ps(W_row[b+1].d * X[b+1].d), acc1);
        }

        for (; b < num_blocks_k; ++b) {
            __m256i d = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(_mm256_loadu_si256((const __m256i*)X[b].qs)), 
                        _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)W_row[b].qs), _mm256_loadu_si256((const __m256i*)X[b].qs))), ones_16);
            acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d), _mm256_set1_ps(W_row[b].d * X[b].d), acc0);
        }

        __m256 final_v = _mm256_add_ps(acc0, acc1);
        __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(final_v), _mm256_extractf128_ps(final_v, 1));
        sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        sum128 = _mm_add_ps(sum128, _mm_shuffle_ps(sum128, sum128, 0x1));
        Y[i] = _mm_cvtss_f32(sum128);
    }
}

void matmul_blocked_int8(int M, int N, int K, const BlockQ8_0* W, const float* X_fp32, float* Y) {
    if (!W || !X_fp32 || !Y) return;
    int nb = K / QK8_0;
    static thread_local std::vector<BlockQ8_0> X_q;
    if (X_q.size() < (size_t)nb) X_q.resize(nb);
    for (int m = 0; m < M; m++) {
        quantize_row_q8_0(X_fp32 + m * K, X_q.data(), K);
        matmul_gemv_int8_avx2(N, K, W, X_q.data(), Y + m * N);
    }
}

void matmul_blocked(int M, int N, int K, const float* A, const float* B, float* C) {
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a = A[i * K + k];
            for (int j = 0; j < N; j++) C[i * N + j] += a * B[k * N + j];
        }
    }
}