#include <cmath>
#include <iostream>
#include <iomanip>
#include "flash_attention.h"
#include <cstring>
#include <omp.h>
#include <chrono>
#include <algorithm>
#include <x86intrin.h>

// ================================================================
// FLASH ATTENTION - Evita materializar QK^T completo
// ================================================================
// MEMORY: seq=2048, dim=64
//   Tradicional: 2048*2048*4 = 16 MB
//   Flash: 64*64*4 = 16 KB
//   REDUÇÃO: 1000x
// ================================================================

void flash_attention(float* O, const float* Q, const float* K, const float* V, int seq, int dim) {
    float scale = 1.0f / sqrtf((float)dim);
    
    #pragma omp parallel for schedule(dynamic, 1)
    for (int qi = 0; qi < seq; qi++) {
        const float* q = Q + qi * dim;
        float* o = O + qi * dim;
        float acc[128] = {0};
        float row_max = -1e9f, row_sum = 0.0f;
        
        for (int kj = 0; kj < seq; kj += FLASH_BLOCK) {
            int k_max = std::min(kj + FLASH_BLOCK, seq);
            float scores[FLASH_BLOCK], weights[FLASH_BLOCK];
            float block_max = -1e9f;
            
            for (int ki = kj; ki < k_max; ki++) {
                float s = 0.0f;
                int d = 0;
                for (; d + 7 < dim; d += 8) {
                    __m256 vq = _mm256_loadu_ps(q + d);
                    __m256 vk = _mm256_loadu_ps(K + ki * dim + d);
                    __m256 vs = _mm256_mul_ps(vq, vk);
                    __m128 lo = _mm256_castps256_ps128(vs);
                    __m128 hi = _mm256_extractf128_ps(vs, 1);
                    lo = _mm_add_ps(lo, hi);
                    lo = _mm_hadd_ps(lo, lo);
                    lo = _mm_hadd_ps(lo, lo);
                    s += _mm_cvtss_f32(lo);
                }
                for (; d < dim; d++) s += q[d] * K[ki * dim + d];
                scores[ki - kj] = s * scale;
                block_max = std::max(block_max, scores[ki - kj]);
            }
            
            float block_sum = 0.0f;
            for (int i = 0; i < k_max - kj; i++) {
                weights[i] = expf(scores[i] - block_max);
                block_sum += weights[i];
            }
            
            float new_max = std::max(row_max, block_max);
            float rescale = expf(row_max - new_max);
            for (int d = 0; d < dim; d++) acc[d] *= rescale;
            
            for (int ki = kj; ki < k_max; ki++) {
                float w = weights[ki - kj] / block_sum;
                for (int d = 0; d < dim; d++) acc[d] += w * V[ki * dim + d];
            }
            
            row_max = new_max;
            row_sum = row_sum * expf(row_max - new_max) + block_sum * expf(block_max - new_max);
        }
        
        for (int d = 0; d < dim; d++) o[d] = acc[d] / row_sum;
    }
}

void benchmark_flash(int seq, int dim) {
    std::cout << "\n╔═══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║         FLASH ATTENTION: seq=" << seq << " dim=" << dim << "                    ║" << std::endl;
    std::cout << "╠═══════════════════════════════════════════════════════════════╣" << std::endl;
    
    long trad_mem = (long)seq * seq * 4;
    long flash_mem = FLASH_BLOCK * FLASH_BLOCK * 4;
    
    std::cout << "║ MEMORY: Trad=" << (trad_mem/1024) << "KB  Flash=" << (flash_mem/1024) << "KB  Savings=" << (trad_mem/flash_mem) << "x ║" << std::endl;
    
    float *Q = (float*)_mm_malloc(seq * dim * 4, 32);
    float *K = (float*)_mm_malloc(seq * dim * 4, 32);
    float *V = (float*)_mm_malloc(seq * dim * 4, 32);
    float *O = (float*)_mm_malloc(seq * dim * 4, 32);
    
    srand(42);
    for (int i = 0; i < seq * dim; i++) {
        Q[i] = (rand() % 100) / 100.0f;
        K[i] = (rand() % 100) / 100.0f;
        V[i] = (rand() % 100) / 100.0f;
    }
    
    flash_attention(O, Q, K, V, seq, dim);
    
    const int runs = 10;
    double total = 0;
    for (int i = 0; i < runs; i++) {
        auto s = std::chrono::high_resolution_clock::now();
        flash_attention(O, Q, K, V, seq, dim);
        total += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - s).count();
    }
    
    double ms = total / runs;
    double gflops = 2.0 * seq * seq * dim / ms / 1e6;
    
    std::cout << "║ TIME: " << std::fixed << std::setprecision(2) << std::setw(8) << ms << " ms  GFLOPS: " << std::setw(8) << gflops << std::endl;
    
    float sum = 0;
    for (int i = 0; i < seq * dim; i++) sum += O[i];
    std::cout << "║ VALID: " << sum << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════╝" << std::endl;
    
    _mm_free(Q); _mm_free(K); _mm_free(V); _mm_free(O);
}
