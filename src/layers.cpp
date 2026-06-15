#include "layers.h"
#include "matmul_blocked.h"
#include "tensor_int8.h"
#include <immintrin.h>
#include <cmath>
#include <cstring>
#include <omp.h>
#include <vector>
#include <iostream>

struct Model;
#include "model.h"

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

void quantize_tensor_to_int8(Tensor& t) {
    if (t.data.empty()) return;
    int total = t.rows * t.cols;
    t.q_data.resize(total / QK8_0);
    quantize_row_q8_0(t.data.data(), t.q_data.data(), total);
    t.data.clear(); t.data.shrink_to_fit();
}

void quantize_model_weights(Model& m) {
    std::cout << "[Step 2] Quantizando para INT8 (2x Unroll)..." << std::endl;
    for (int i = 0; i < m.n_layers; i++) {
        quantize_tensor_to_int8(m.layers[i].wq); quantize_tensor_to_int8(m.layers[i].wk);
        quantize_tensor_to_int8(m.layers[i].wv); quantize_tensor_to_int8(m.layers[i].wo);
        quantize_tensor_to_int8(m.layers[i].w1); quantize_tensor_to_int8(m.layers[i].w2);
        quantize_tensor_to_int8(m.layers[i].w3);
    }
    quantize_tensor_to_int8(m.lm_head);
}

void attention_forward(Tensor& out, const Tensor& x, const Tensor& wq, const Tensor& wk, const Tensor& wv, const Tensor& wo, KVCache& cache, int pos, int dim, int n_heads) {
    int head_dim = dim / n_heads;
    std::vector<float> q(dim), k_v(dim), v_v(dim);
    matmul_blocked_int8(1, dim, dim, wq.q_data.data(), x.data.data(), q.data());
    matmul_blocked_int8(1, dim, dim, wk.q_data.data(), x.data.data(), k_v.data());
    matmul_blocked_int8(1, dim, dim, wv.q_data.data(), x.data.data(), v_v.data());
    apply_rope_raw(q.data(), pos, dim);
    apply_rope_raw(k_v.data(), pos, dim);
    std::memcpy(&cache.k_cache[pos * dim], k_v.data(), dim * sizeof(float));
    std::memcpy(&cache.v_cache[pos * dim], v_v.data(), dim * sizeof(float));
    std::vector<float> attn_out_vec(dim, 0.0f);
    float scale = 1.0f / sqrtf((float)head_dim);
    #pragma omp parallel for
    for (int h = 0; h < n_heads; h++) {
        std::vector<float> scores(pos + 1);
        float* q_head = &q[h * head_dim];
        for (int t = 0; t <= pos; t++) {
            float sum = 0;
            float* k_head = &cache.k_cache[t * dim + h * head_dim];
            for (int d = 0; d < head_dim; d++) sum += q_head[d] * k_head[d];
            scores[t] = sum * scale;
        }
        softmax(scores.data(), pos + 1);
        float* out_head = &attn_out_vec[h * head_dim];
        for (int t = 0; t <= pos; t++) {
            float* v_head = &cache.v_cache[t * dim + h * head_dim];
            for (int d = 0; d < head_dim; d++) out_head[d] += scores[t] * v_head[d];
        }
    }
    matmul_blocked_int8(1, dim, dim, wo.q_data.data(), attn_out_vec.data(), out.data.data());
}

void ffn_forward(Tensor& out, const Tensor& x, const Tensor& w1, const Tensor& w2, const Tensor& w3, int dim, int hidden_dim) {
    std::vector<float> g(hidden_dim), u(hidden_dim);
    matmul_blocked_int8(1, hidden_dim, dim, w1.q_data.data(), x.data.data(), g.data());
    matmul_blocked_int8(1, hidden_dim, dim, w3.q_data.data(), x.data.data(), u.data());
    for (int i = 0; i < hidden_dim; i++) g[i] = (g[i] / (1.0f + expf(-g[i]))) * u[i];
    matmul_blocked_int8(1, dim, hidden_dim, w2.q_data.data(), g.data(), out.data.data());
}