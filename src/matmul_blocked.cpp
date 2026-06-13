#include "matmul_blocked.h"
#include "kernel_avx2.h"
#include <algorithm>
#include <cstring>
#include <omp.h>

void matmul_blocked(int M, int N, int K, const float* A, const float* B, float* C) {
    // 1. Zera a matriz C
    std::memset(C, 0, (size_t)M * N * sizeof(float));

    // 2. Tiles ajustados para o seu Ryzen (L2=512KB, L3=32MB)
    const int mc = 96; 
    const int nc = 256;
    const int kc = 256;

    // 3. Paralelismo nos eixos de saída I e J (Race-free)
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < M; i += mc) {
        for (int j = 0; j < N; j += nc) {
            
            // Loop de K interno: Cada thread processa toda a profundidade K para seu bloco de C
            for (int k = 0; k < K; k += kc) {
                int k_len = std::min(kc, K - k);
                int i_limit = std::min(i + mc, M);
                int j_limit = std::min(j + nc, N);

                // Loops do Micro-kernel
                for (int ii = i; ii < i_limit; ii += 6) {
                    for (int jj = j; jj < j_limit; jj += 16) {
                        
                        if (ii + 6 <= i_limit && jj + 16 <= j_limit) {
                            // Chama o micro-kernel acumulativo
                            micro_kernel_6x16_acc(k_len, &A[ii * K + k], K, &B[k * N + jj], N, &C[ii * N + jj], N);
                        } else {
                            // Fallback de bordas para garantir o "OK"
                            for (int r = ii; r < std::min(ii + 6, i_limit); ++r) {
                                for (int c = jj; c < std::min(jj + 16, j_limit); ++c) {
                                    float sum = 0;
                                    for (int kk = k; kk < k + k_len; ++kk) {
                                        sum += A[r * K + kk] * B[kk * N + c];
                                    }
                                    C[r * N + c] += sum;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}