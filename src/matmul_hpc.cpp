#include <iostream>
#include <iomanip>
#include "matmul_hpc.h"
#include <cstring>
#include <omp.h>
#include <chrono>
#include <algorithm>
#include <x86intrin.h>

// NAIVE: i,j,k - mau uso de cache
void matmul_naive(float* C, const float* A, const float* B, int M, int N, int K) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// IKJ: melhor acesso memória
void matmul_ikj(float* C, const float* A, const float* B, int M, int N, int K) {
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = A[i * K + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

// TILED: cache blocking
void matmul_tiled(float* C, const float* A, const float* B, int M, int N, int K) {
    memset(C, 0, M * N * sizeof(float));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i += BM) {
        for (int j = 0; j < N; j += BN) {
            for (int k = 0; k < K; k += BK) {
                int i_max = std::min(i + BM, M);
                int j_max = std::min(j + BN, N);
                int k_max = std::min(k + BK, K);
                for (int ii = i; ii < i_max; ii++) {
                    for (int kk = k; kk < k_max; kk++) {
                        float a = A[ii * K + kk];
                        int b_off = kk * N + j;
                        int c_off = ii * N + j;
                        for (int jj = j; jj < j_max; jj += 4) {
                            __m128 va = _mm_set1_ps(a);
                            __m128 vc = _mm_loadu_ps(&C[c_off + jj]);
                            __m128 vb = _mm_loadu_ps(&B[b_off + jj]);
                            vc = _mm_fmadd_ps(va, vb, vc);
                            _mm_storeu_ps(&C[c_off + jj], vc);
                        }
                    }
                }
            }
        }
    }
}

// SIMD: AVX2 completo
void matmul_simd(float* C, const float* A, const float* B, int M, int N, int K) {
    memset(C, 0, M * N * sizeof(float));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i += BM) {
        for (int j = 0; j < N; j += BN) {
            for (int k = 0; k < K; k += BK) {
                int i_max = std::min(i + BM, M);
                int j_max = std::min(j + BN, N);
                int k_max = std::min(k + BK, K);
                for (int ii = i; ii < i_max; ii++) {
                    for (int kk = k; kk < k_max; kk++) {
                        float a = A[ii * K + kk];
                        __m256 va = _mm256_set1_ps(a);
                        int b_base = kk * N + j;
                        int c_base = ii * N + j;
                        int jj = j;
                        for (; jj + 7 < j_max; jj += 8) {
                            __m256 vb = _mm256_loadu_ps(&B[b_base + jj]);
                            __m256 vc = _mm256_loadu_ps(&C[c_base + jj]);
                            vc = _mm256_fmadd_ps(va, vb, vc);
                            _mm256_storeu_ps(&C[c_base + jj], vc);
                        }
                        for (; jj < j_max; jj++) {
                            C[c_base + jj] += a * B[b_base + jj];
                        }
                    }
                }
            }
        }
    }
}

void benchmark_matmul(int M, int N, int K) {
    std::cout << "\n╔═══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║         MATMUL: " << M << "x" << N << "x" << K << "                        ║" << std::endl;
    std::cout << "╠═══════════════════════════════════════════════════════════════╣" << std::endl;
    
    float *A = (float*)_mm_malloc(M * K * sizeof(float), 32);
    float *B = (float*)_mm_malloc(K * N * sizeof(float), 32);
    float *C = (float*)_mm_malloc(M * N * sizeof(float), 32);
    
    srand(42);
    for (int i = 0; i < M * K; i++) A[i] = (rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) B[i] = (rand() % 100) / 100.0f;
    
    auto run = [&](void(*fn)(float*, const float*, const float*, int, int, int), const char* name, float& t) {
        const int runs = 5;
        double total = 0;
        memset(C, 0, M * N * sizeof(float));
        fn(C, A, B, M, N, K);
        for (int i = 0; i < runs; i++) {
            memset(C, 0, M * N * sizeof(float));
            auto s = std::chrono::high_resolution_clock::now();
            fn(C, A, B, M, N, K);
            total += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - s).count();
        }
        t = total / runs;
        double g = 2.0 * M * N * K / t / 1e6;
        std::cout << "║ " << std::setw(16) << name << ": " << std::fixed << std::setprecision(2)
                  << std::setw(8) << t << " ms | " << std::setw(8) << g << " GFLOPS ║" << std::endl;
    };
    
    float t_n, t_ikj, t_t, t_s;
    run(matmul_naive, "NAIVE (baseline)", t_n);
    run(matmul_ikj, "IKJ", t_ikj);
    run(matmul_tiled, "TILED", t_t);
    run(matmul_simd, "SIMD (AVX2)", t_s);
    
    std::cout << "╠═══════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║ SPEEDUP: IKJ=" << std::fixed << std::setprecision(1) << t_n/t_ikj << "x  TILED=" << t_n/t_t << "x  SIMD=" << t_n/t_s << "x" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════╝" << std::endl;
    
    _mm_free(A); _mm_free(B); _mm_free(C);
}
