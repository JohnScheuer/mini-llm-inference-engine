#include "layers.h"
#include "matmul_blocked.h"
#include <cmath>
#include <cstdio>
#include <vector>
#include <omp.h>
#include <algorithm>
#include <immintrin.h>

// =====================================================
//  Kernels de Baixo Nível (Vetorização AVX2 + FMA)
// =====================================================

// Soma horizontal de um registrador AVX (__m256 -> float)
inline float _mm256_reduce_add_ps(__m256 x) {
    __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    return _mm_cvtss_f32(x32);
}

// Produto Escalar (Dot Product) Otimizado
float dot_product_avx(const float* a, const float* b, int n) {
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    float res = _mm256_reduce_add_ps(sum);
    for (; i < n; i++) res += a[i] * b[i];
    return res;
}

// =====================================================
//  Implementação das Camadas Principais (Engine)
// =====================================================

void rmsnorm(Tensor& out, const Tensor& x, const Tensor& gamma, float eps) {
    float norm = 0.0f;
    int dim = x.cols;
    for (int i = 0; i < dim; i++) norm += x.data[i] * x.data[i];
    norm = sqrtf(norm / dim + eps);
    for (int i = 0; i < dim; i++) {
        out.data[i] = (x.data[i] / norm) * gamma.data[i];
    }
}

void apply_rope_raw(float* data, int pos, int dim) {
    for (int i = 0; i < dim; i += 2) {
        float freq = 1.0f / powf(10000.0f, (float)i / dim);
        float val = (float)pos * freq;
        float cos_val = cosf(val);
        float sin_val = sinf(val);
        float x0 = data[i];
        float x1 = data[i + 1];
        data[i]     = x0 * cos_val - x1 * sin_val;
        data[i + 1] = x0 * sin_val + x1 * cos_val;
    }
}

// Wrapper para o seu kernel de alta performance
void matmul(Tensor& out, const Tensor& a, const Tensor& b) {
    matmul_blocked(a.rows, b.cols, a.cols, (float*)a.data.data(), (float*)b.data.data(), (float*)out.data.data());
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

void attention_forward(
    Tensor& out, const Tensor& x,
    const Tensor& wq, const Tensor& wk, const Tensor& wv, const Tensor& wo,
    KVCache& cache, int pos, int dim, int n_heads
) {
    int head_dim = dim / n_heads;
    float scale = 1.0f / sqrtf((float)head_dim);

    Tensor q(1, dim), k(1, dim), v(1, dim);
    matmul(q, x, wq);
    matmul(k, x, wk);
    matmul(v, x, wv);

    for (int h = 0; h < n_heads; h++) {
        apply_rope_raw(&q.data[h * head_dim], pos, head_dim);
        apply_rope_raw(&k.data[h * head_dim], pos, head_dim);
    }

    // Salva K e V no cache
    for (int i = 0; i < dim; i++) {
        cache.k_cache[pos * dim + i] = k.data[i];
        cache.v_cache[pos * dim + i] = v.data[i];
    }

    Tensor attn_out(1, dim);
    #pragma omp parallel for
    for (int h = 0; h < n_heads; h++) {
        std::vector<float> scores(pos + 1);
        const float* q_ptr = &q.data[h * head_dim];

        // 1. Attention Scores (Query * Key) - VETORIZADO
        for (int t = 0; t <= pos; t++) {
            const float* k_cached = &cache.k_cache[t * dim + h * head_dim];
            scores[t] = dot_product_avx(q_ptr, k_cached, head_dim) * scale;
        }

        // 2. Softmax
        softmax(scores.data(), pos + 1);

        // 3. Agregação de Values (Score * Value) - VETORIZADO ✨
        float* out_ptr = &attn_out.data[h * head_dim];
        // Zera o buffer de saída da cabeça
        for(int i=0; i<head_dim; i++) out_ptr[i] = 0;

        for (int t = 0; t <= pos; t++) {
            __m256 s = _mm256_set1_ps(scores[t]);
            const float* v_cached = &cache.v_cache[t * dim + h * head_dim];
            
            int i = 0;
            for (; i <= head_dim - 8; i += 8) {
                __m256 v_vec = _mm256_loadu_ps(&v_cached[i]);
                __m256 acc = _mm256_loadu_ps(&out_ptr[i]);
                acc = _mm256_fmadd_ps(s, v_vec, acc);
                _mm256_storeu_ps(&out_ptr[i], acc);
            }
            // Tail loop para head_dim não múltiplo de 8
            for (; i < head_dim; i++) {
                out_ptr[i] += scores[t] * v_cached[i];
            }
        }
    }
    // Projeção de saída
    matmul(out, attn_out, wo);
}

float silu(float x) { return x / (1.0f + expf(-x)); }

void ffn_forward(Tensor& out, const Tensor& x, const Tensor& w1, const Tensor& w2, const Tensor& w3, int dim, int hidden_dim) {
    Tensor gate(1, hidden_dim), up(1, hidden_dim), combined(1, hidden_dim);
    matmul(gate, x, w1);
    #pragma omp parallel for
    for (int i = 0; i < hidden_dim; i++) gate.data[i] = silu(gate.data[i]);
    matmul(up, x, w3);
    #pragma omp parallel for
    for (int i = 0; i < hidden_dim; i++) combined.data[i] = gate.data[i] * up.data[i];
    matmul(out, combined, w2);
}

// =====================================================
//  MatMul: Versões Legadas (Para o Benchmark)
// =====================================================

void matmul_naive(Tensor& C, const Tensor& A, const Tensor& B) {
    int M = A.rows, K = A.cols, N = B.cols;
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += A.at(i, k) * B.at(k, j);
            C.at(i, j) = sum;
        }
    }
}

void matmul_ikj(Tensor& C, const Tensor& A, const Tensor& B) {
    int M = A.rows, K = A.cols, N = B.cols;
    std::fill(C.data.begin(), C.data.end(), 0.0f);
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = A.at(i, k);
            for (int j = 0; j < N; j++) C.data[i * N + j] += a_ik * B.data[k * N + j];
        }
    }
}

void matmul_tiled_ikj(Tensor& C, const Tensor& A, const Tensor& B, int T) {
    int M = A.rows, K = A.cols, N = B.cols;
    std::fill(C.data.begin(), C.data.end(), 0.0f);
    #pragma omp parallel for collapse(2)
    for (int i0 = 0; i0 < M; i0 += T) {
        for (int j0 = 0; j0 < N; j0 += T) {
            for (int k0 = 0; k0 < K; k0 += T) {
                for (int i = i0; i < std::min(i0+T, M); i++) {
                    for (int k = k0; k < std::min(k0+T, K); k++) {
                        float a_ik = A.at(i, k);
                        for (int j = j0; j < std::min(j0+T, N); j++) C.data[i * N + j] += a_ik * B.data[k * N + j];
                    }
                }
            }
        }
    }
}

void matmul_ikj_unroll4(Tensor& C, const Tensor& A, const Tensor& B) {
    int M = A.rows, K = A.cols, N = B.cols;
    std::fill(C.data.begin(), C.data.end(), 0.0f);
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = A.at(i, k);
            int j = 0;
            for (; j + 4 <= N; j += 4) {
                C.data[i * N + j] += a_ik * B.data[k * N + j];
                C.data[i * N + j+1] += a_ik * B.data[k * N + j+1];
                C.data[i * N + j+2] += a_ik * B.data[k * N + j+2];
                C.data[i * N + j+3] += a_ik * B.data[k * N + j+3];
            }
            for (; j < N; j++) C.data[i * N + j] += a_ik * B.data[k * N + j];
        }
    }
}

void matmul_tiled_unroll4(Tensor& C, const Tensor& A, const Tensor& B, int T) {
    int M = A.rows, K = A.cols, N = B.cols;
    std::fill(C.data.begin(), C.data.end(), 0.0f);
    #pragma omp parallel for collapse(2)
    for (int i0 = 0; i0 < M; i0 += T) {
        for (int j0 = 0; j0 < N; j0 += T) {
            for (int k0 = 0; k0 < K; k0 += T) {
                for (int i = i0; i < std::min(i0+T, M); i++) {
                    for (int k = k0; k < std::min(k0+T, K); k++) {
                        float a_ik = A.at(i, k);
                        int j = j0;
                        for (; j + 4 <= std::min(j0+T, N); j += 4) {
                            C.data[i * N + j] += a_ik * B.data[k * N + j];
                            C.data[i * N + j+1] += a_ik * B.data[k * N + j+1];
                            C.data[i * N + j+2] += a_ik * B.data[k * N + j+2];
                            C.data[i * N + j+3] += a_ik * B.data[k * N + j+3];
                        }
                        for (; j < std::min(j0+T, N); j++) C.data[i * N + j] += a_ik * B.data[k * N + j];
                    }
                }
            }
        }
    }
}

void matmul_tiled_unroll8(Tensor& C, const Tensor& A, const Tensor& B, int T) {
    int M = A.rows, K = A.cols, N = B.cols;
    std::fill(C.data.begin(), C.data.end(), 0.0f);
    #pragma omp parallel for collapse(2)
    for (int i0 = 0; i0 < M; i0 += T) {
        for (int j0 = 0; j0 < N; j0 += T) {
            for (int k0 = 0; k0 < K; k0 += T) {
                for (int i = i0; i < std::min(i0+T, M); i++) {
                    int j = j0;
                    for (; j + 8 <= std::min(j0+T, N); j += 8) {
                        float c[8] = {0};
                        for (int k = k0; k < std::min(k0+T, K); k++) {
                            float a = A.at(i, k);
                            for(int jj=0; jj<8; jj++) c[jj] += a * B.data[k * N + j + jj];
                        }
                        for(int jj=0; jj<8; jj++) C.data[i * N + j + jj] += c[jj];
                    }
                    for (; j < std::min(j0+T, N); j++) {
                        for (int k = k0; k < std::min(k0+T, K); k++) C.data[i * N + j] += A.at(i, k) * B.data[k * N + j];
                    }
                }
            }
        }
    }
}