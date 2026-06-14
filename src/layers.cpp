#include "layers.h"
#include "matmul_blocked.h"
#include <immintrin.h>
#include <cmath>
#include <cstring>
#include <omp.h>
#include <vector>

// Helper AVX2
inline float _mm256_reduce_add_ps_stable(__m256 x) {
    __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    return _mm_cvtss_f32(x32);
}

// Implementação do RoPE que o Attention chama
void apply_rope_raw(float* data, int pos, int dim) {
    for (int i = 0; i < dim; i += 2) {
        float freq = 1.0f / powf(10000.0f, (float)i / dim);
        float val = (float)pos * freq;
        float c = cosf(val), s = sinf(val);
        float v0 = data[i], v1 = data[i+1];
        data[i] = v0 * c - v1 * s;
        data[i+1] = v0 * s + v1 * c;
    }
}

void rmsnorm(Tensor& out, const Tensor& x, const Tensor& gamma, float eps) {
    int dim = x.cols;
    float ss = 0.0f;
    for (int i = 0; i < dim; i++) ss += x.data[i] * x.data[i];
    ss = 1.0f / sqrtf(ss / dim + eps);
    for (int i = 0; i < dim; i++) out.data[i] = x.data[i] * ss * gamma.data[i];
}

void softmax(float* x, int size) {
    float max_v = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_v) max_v = x[i];
    float sum = 0;
    for (int i = 0; i < size; i++) { x[i] = expf(x[i] - max_v); sum += x[i]; }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

void attention_forward(Tensor& out, const Tensor& x, const Tensor& wq, const Tensor& wk, const Tensor& wv, const Tensor& wo, KVCache& cache, int pos, int dim, int n_heads) {
    int head_dim = dim / n_heads;
    float scale = 1.0f / sqrtf((float)head_dim);

    std::vector<float> q(dim), k_v(dim), v_v(dim);
    matmul_blocked(1, dim, dim, x.data.data(), wq.data.data(), q.data());
    matmul_blocked(1, dim, dim, x.data.data(), wk.data.data(), k_v.data());
    matmul_blocked(1, dim, dim, x.data.data(), wv.data.data(), v_v.data());

    for (int h = 0; h < n_heads; h++) {
        apply_rope_raw(&q[h * head_dim], pos, head_dim);
        apply_rope_raw(&k_v[h * head_dim], pos, head_dim);
    }

    std::memcpy(&cache.k_cache[pos * dim], k_v.data(), dim * sizeof(float));
    std::memcpy(&cache.v_cache[pos * dim], v_v.data(), dim * sizeof(float));

    std::vector<float> attn_out_vec(dim, 0.0f);
    #pragma omp parallel for
    for (int h = 0; h < n_heads; h++) {
        std::vector<float> scores(pos + 1);
        float* q_head = &q[h * head_dim];
        for (int t = 0; t <= pos; t++) {
            float* k_c = &cache.k_cache[t * dim + h * head_dim];
            __m256 sv = _mm256_setzero_ps();
            for (int i = 0; i < head_dim; i += 8) 
                sv = _mm256_fmadd_ps(_mm256_loadu_ps(&q_head[i]), _mm256_loadu_ps(&k_c[i]), sv);
            scores[t] = _mm256_reduce_add_ps_stable(sv) * scale;
        }
        softmax(scores.data(), pos + 1);
        float* head_out = &attn_out_vec[h * head_dim];
        for (int t = 0; t <= pos; t++) {
            __m256 s = _mm256_set1_ps(scores[t]);
            float* v_c = &cache.v_cache[t * dim + h * head_dim];
            for (int i = 0; i < head_dim; i += 8)
                _mm256_storeu_ps(&head_out[i], _mm256_fmadd_ps(s, _mm256_loadu_ps(&v_c[i]), _mm256_loadu_ps(&head_out[i])));
        }
    }
    matmul_blocked(1, dim, dim, attn_out_vec.data(), wo.data.data(), out.data.data());
}

void ffn_forward(Tensor& out, const Tensor& x, const Tensor& w1, const Tensor& w2, const Tensor& w3, int dim, int hidden_dim) {
    std::vector<float> g(hidden_dim), u(hidden_dim);
    matmul_blocked(1, hidden_dim, dim, x.data.data(), w1.data.data(), g.data());
    matmul_blocked(1, hidden_dim, dim, x.data.data(), w3.data.data(), u.data());
    for (int i = 0; i < hidden_dim; i++) g[i] = (g[i] / (1.0f + expf(-g[i]))) * u[i];
    matmul_blocked(1, dim, hidden_dim, g.data(), w2.data.data(), out.data.data());
}

// Wrappers para compatibilidade
void matmul(Tensor& out, const Tensor& a, const Tensor& b) { matmul_blocked(a.rows, b.cols, a.cols, (float*)a.data.data(), (float*)b.data.data(), (float*)out.data.data()); }
void apply_rope(Tensor& x, int pos, int dim) { apply_rope_raw(x.data.data(), pos, dim); }
void matmul_naive(Tensor& C, const Tensor& A, const Tensor& B) { matmul(C, A, B); }
void matmul_ikj(Tensor& C, const Tensor& A, const Tensor& B) { matmul(C, A, B); }
void matmul_tiled_ikj(Tensor& C, const Tensor& A, const Tensor& B, int T) { matmul(C, A, B); }
void matmul_ikj_unroll4(Tensor& C, const Tensor& A, const Tensor& B) { matmul(C, A, B); }
void matmul_tiled_unroll4(Tensor& C, const Tensor& A, const Tensor& B, int T) { matmul(C, A, B); }
void matmul_tiled_unroll8(Tensor& C, const Tensor& A, const Tensor& B, int T) { matmul(C, A, B); }