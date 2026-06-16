#include "layers.h"
#include "matmul_blocked.h"
#include <immintrin.h>
#include <cmath>
#include <cstring>
#include <omp.h>

float dot_product_avx2(const float* a, const float* b, int size) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    int d = 0;
    for (; d <= size - 16; d += 16) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + d),
                               _mm256_loadu_ps(b + d), sum0);
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + d + 8),
                               _mm256_loadu_ps(b + d + 8), sum1);
    }
    for (; d <= size - 8; d += 8) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + d),
                               _mm256_loadu_ps(b + d), sum0);
    }
    sum0 = _mm256_add_ps(sum0, sum1);
    __m128 s128 = _mm_add_ps(_mm256_castps256_ps128(sum0),
                              _mm256_extractf128_ps(sum0, 1));
    s128 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
    s128 = _mm_add_ps(s128, _mm_shuffle_ps(s128, s128, 0x1));
    float result = _mm_cvtss_f32(s128);
    for (; d < size; d++) result += a[d] * b[d];
    return result;
}

void rmsnorm(float* o, const float* x, const float* gamma,
             int dim, float eps) {
    __m256 ss_vec = _mm256_setzero_ps();
    int i = 0;
    for (; i <= dim - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        ss_vec = _mm256_fmadd_ps(v, v, ss_vec);
    }
    __m128 s128 = _mm_add_ps(_mm256_castps256_ps128(ss_vec),
                              _mm256_extractf128_ps(ss_vec, 1));
    s128 = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
    s128 = _mm_add_ps(s128, _mm_shuffle_ps(s128, s128, 0x1));
    float ss = _mm_cvtss_f32(s128);
    for (; i < dim; i++) ss += x[i] * x[i];

    float scale = 1.0f / sqrtf(ss / dim + eps);
    __m256 scale_vec = _mm256_set1_ps(scale);

    i = 0;
    for (; i <= dim - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 g = _mm256_loadu_ps(gamma + i);
        _mm256_storeu_ps(o + i,
            _mm256_mul_ps(_mm256_mul_ps(v, scale_vec), g));
    }
    for (; i < dim; i++) o[i] = x[i] * scale * gamma[i];
}

void softmax(float* x, int size) {
    float max_v = x[0];
    for (int i = 1; i < size; i++)
        if (x[i] > max_v) max_v = x[i];
    float sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_v);
        sum += x[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < size; i++) x[i] *= inv_sum;
}

void apply_rope_precomputed(float* data, const float* cos_buf,
                            const float* sin_buf, int dim) {
    for (int i = 0; i < dim; i += 2) {
        float v0 = data[i], v1 = data[i + 1];
        data[i]     = v0 * cos_buf[i] - v1 * sin_buf[i];
        data[i + 1] = v0 * sin_buf[i] + v1 * cos_buf[i];
    }
}

void attention_forward(float* attn_out, const float* x,
                       const Tensor& wq, const Tensor& wk,
                       const Tensor& wv, const Tensor& wo,
                       KVCache& cache,
                       float* q_buf, float* k_buf,
                       float* v_buf, float* scores_buf,
                       const float* rope_cos, const float* rope_sin,
                       int pos, int dim, int n_heads,
                       int max_seq_len, BlockQ8_0* xq_buf) {

    int head_dim = dim / n_heads;

    // Quantiza x UMA VEZ, reutiliza para wq, wk, wv
    int nb = dim / QK8_0;
    quantize_row_q8_0(x, xq_buf, dim);

    matmul_gemv_int8_avx2(dim, dim, wq.q_data.data(), xq_buf, q_buf);
    matmul_gemv_int8_avx2(dim, dim, wk.q_data.data(), xq_buf, k_buf);
    matmul_gemv_int8_avx2(dim, dim, wv.q_data.data(), xq_buf, v_buf);

    apply_rope_precomputed(q_buf, rope_cos, rope_sin, dim);
    apply_rope_precomputed(k_buf, rope_cos, rope_sin, dim);

    std::memcpy(&cache.k_cache[pos * dim], k_buf, dim * sizeof(float));
    std::memcpy(&cache.v_cache[pos * dim], v_buf, dim * sizeof(float));

    float* pre_wo = v_buf;
    std::memset(pre_wo, 0, dim * sizeof(float));

    float scale = 1.0f / sqrtf((float)head_dim);

    #pragma omp parallel for if(n_heads > 16)
    for (int h = 0; h < n_heads; h++) {
        float* q_head = q_buf + h * head_dim;
        float* scores_head = scores_buf + h * max_seq_len;

        for (int t = 0; t <= pos; t++) {
            float* k_head = &cache.k_cache[t * dim + h * head_dim];
            scores_head[t] = dot_product_avx2(q_head, k_head, head_dim) * scale;
        }

        softmax(scores_head, pos + 1);

        float* out_head = pre_wo + h * head_dim;
        for (int t = 0; t <= pos; t++) {
            float* v_head = &cache.v_cache[t * dim + h * head_dim];
            float score = scores_head[t];
            __m256 sv = _mm256_set1_ps(score);
            int d = 0;
            for (; d <= head_dim - 8; d += 8) {
                __m256 vv = _mm256_loadu_ps(v_head + d);
                __m256 ov = _mm256_loadu_ps(out_head + d);
                _mm256_storeu_ps(out_head + d,
                    _mm256_fmadd_ps(sv, vv, ov));
            }
            for (; d < head_dim; d++)
                out_head[d] += score * v_head[d];
        }
    }

    // Projeção Wo: requantiza pre_wo (diferente de x, então precisa)
    matmul_blocked_int8(1, dim, dim, wo.q_data.data(), pre_wo, attn_out, xq_buf);
}

void ffn_forward(float* out, const float* x,
                 const Tensor& w1, const Tensor& w2, const Tensor& w3,
                 float* ffn_g_buf, float* ffn_u_buf,
                 int dim, int hidden_dim, BlockQ8_0* xq_buf) {

    // OTIMIZAÇÃO CHAVE: Quantiza x UMA VEZ, reutiliza para w1 e w3
    quantize_row_q8_0(x, xq_buf, dim);
    matmul_gemv_int8_avx2(hidden_dim, dim, w1.q_data.data(), xq_buf, ffn_g_buf);
    matmul_gemv_int8_avx2(hidden_dim, dim, w3.q_data.data(), xq_buf, ffn_u_buf);

    // SwiGLU
    for (int i = 0; i < hidden_dim; i++) {
        float g = ffn_g_buf[i];
        ffn_g_buf[i] = (g / (1.0f + expf(-g))) * ffn_u_buf[i];
    }

    // w2: input é ffn_g_buf (hidden_dim), precisa requantizar
    matmul_blocked_int8(1, dim, hidden_dim, w2.q_data.data(), ffn_g_buf, out, xq_buf);
}

void quantize_tensor_to_int8(Tensor& t) {
    if (t.data.empty()) return;
    int total = t.rows * t.cols;
    t.q_data.resize(total / QK8_0);
    quantize_row_q8_0(t.data.data(), t.q_data.data(), total);
    t.data.clear();
    t.data.shrink_to_fit();
}

void quantize_model_weights(Model& m) {
    for (int i = 0; i < m.n_layers; i++) {
        quantize_tensor_to_int8(m.layers[i].wq);
        quantize_tensor_to_int8(m.layers[i].wk);
        quantize_tensor_to_int8(m.layers[i].wv);
        quantize_tensor_to_int8(m.layers[i].wo);
        quantize_tensor_to_int8(m.layers[i].w1);
        quantize_tensor_to_int8(m.layers[i].w2);
        quantize_tensor_to_int8(m.layers[i].w3);
    }
    quantize_tensor_to_int8(m.lm_head);
}