#include "matmul_blocked.h"
#include "kernel_avx2.h"
#include <algorithm>
#include <cstring>
#include <omp.h>

void matmul_blocked(int M, int N, int K, const float* A, const float* B, float* C) {
    std::memset(C, 0, (size_t)M * N * sizeof(float));

    // mc=48 cria ~21 blocos para 1024 linhas, ideal para 12 threads
    const int mc = 48;  
    const int nc = 512; 
    const int kc = 256; 

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < M; i += mc) {
        for (int j = 0; j < N; j += nc) {
            
            for (int k = 0; k < K; k += kc) {
                int k_len = std::min(kc, K - k);
                int i_limit = std::min(i + mc, M);
                int j_limit = std::min(j + nc, N);

                for (int ii = i; ii < i_limit; ii += 6) {
                    for (int jj = j; jj < j_limit; jj += 16) {
                        if (ii + 6 <= i_limit && jj + 16 <= j_limit) {
                            micro_kernel_6x16_acc(k_len, &A[ii * K + k], K, &B[k * N + jj], N, &C[ii * N + jj], N);
                        } else {
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