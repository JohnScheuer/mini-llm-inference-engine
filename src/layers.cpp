#include "layers.h"
#include <cmath>
#include <cstdio>
#include <vector>
#include <omp.h>

// RMSNorm
void rmsnorm(Tensor& out, const Tensor& x, const Tensor& gamma, float eps) {
    float norm = 0.0f;
    int dim = x.cols;
    for (int i = 0; i < dim; i++) {
        norm += x.data[i] * x.data[i];
    }
    norm = sqrtf(norm / dim + eps);
    for (int i = 0; i < dim; i++) {
        out.data[i] = (x.data[i] / norm) * gamma.data[i];
    }
}

// RoPE
void apply_rope(Tensor& x, int pos, int dim) {
    for (int i = 0; i < dim; i += 2) {
        float freq = 1.0f / powf(10000.0f, (float)i / dim);
        float val = pos * freq;
        float cos_val = cosf(val);
        float sin_val = sinf(val);

        float x0 = x.data[i];
        float x1 = x.data[i + 1];

        x.data[i]     = x0 * cos_val - x1 * sin_val;
        x.data[i + 1] = x0 * sin_val + x1 * cos_val;
    }
}

// MatMul
void matmul(Tensor& out, const Tensor& a, const Tensor& b) {
    #pragma omp parallel for
    for (int c = 0; c < out.cols; c++) {
        float sum = 0.0f;
        for (int k = 0; k < a.cols; k++) {
            sum += a.at(0, k) * b.at(k, c);
        }
        out.at(0, c) = sum;
    }
}

// Softmax
void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }

    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// Multi-Head Attention com KV-Cache
void attention_forward(
    Tensor& out, const Tensor& x,
    const Tensor& wq, const Tensor& wk, const Tensor& wv, const Tensor& wo,
    KVCache& cache, int pos, int dim, int n_heads
) {
    int head_dim = dim / n_heads;
    float scale = 1.0f / sqrtf((float)head_dim);

    Tensor q(1, dim);
    Tensor k(1, dim);
    Tensor v(1, dim);

    matmul(q, x, wq);
    matmul(k, x, wk);
    matmul(v, x, wv);

    // Aplicar RoPE em cada cabeça
    for (int h = 0; h < n_heads; h++) {
        Tensor q_head(1, head_dim);
        Tensor k_head(1, head_dim);
        
        for (int i = 0; i < head_dim; i++) {
            q_head.data[i] = q.data[h * head_dim + i];
            k_head.data[i] = k.data[h * head_dim + i];
        }
        
        apply_rope(q_head, pos, head_dim);
        apply_rope(k_head, pos, head_dim);
        
        for (int i = 0; i < head_dim; i++) {
            q.data[h * head_dim + i] = q_head.data[i];
            k.data[h * head_dim + i] = k_head.data[i];
        }
    }

    // Salvar K e V no cache
    for (int i = 0; i < dim; i++) {
        cache.k_cache[pos * dim + i] = k.data[i];
        cache.v_cache[pos * dim + i] = v.data[i];
    }

    // Attention por cabeça
    Tensor attn_out(1, dim);
    
    #pragma omp parallel for
    for (int h = 0; h < n_heads; h++) {
        std::vector<float> scores(pos + 1, 0.0f);
        
        for (int t = 0; t <= pos; t++) {
            float score = 0.0f;
            for (int i = 0; i < head_dim; i++) {
                score += q.data[h * head_dim + i] * 
                         cache.k_cache[t * dim + h * head_dim + i];
            }
            scores[t] = score * scale;
        }

        softmax(scores.data(), pos + 1);

        for (int i = 0; i < head_dim; i++) {
            float sum = 0.0f;
            for (int t = 0; t <= pos; t++) {
                sum += scores[t] * cache.v_cache[t * dim + h * head_dim + i];
            }
            attn_out.data[h * head_dim + i] = sum;
        }
    }

    matmul(out, attn_out, wo);
}

// SiLU
float silu(float x) {
    return x / (1.0f + expf(-x));
}

// FFN com SwiGLU
void ffn_forward(
    Tensor& out, const Tensor& x,
    const Tensor& w1, const Tensor& w2, const Tensor& w3,
    int dim, int hidden_dim
) {
    Tensor gate(1, hidden_dim);
    matmul(gate, x, w1);
    
    #pragma omp parallel for
    for (int i = 0; i < hidden_dim; i++) {
        gate.data[i] = silu(gate.data[i]);
    }

    Tensor up(1, hidden_dim);
    matmul(up, x, w3);

    Tensor combined(1, hidden_dim);
    #pragma omp parallel for
    for (int i = 0; i < hidden_dim; i++) {
        combined.data[i] = gate.data[i] * up.data[i];
    }

    matmul(out, combined, w2);
}
// MatMul ingênua genérica - BASELINE
// Loop triplo clássico, ordem i-j-k (RUIM pra cache, mas é o padrão)
void matmul_naive(Tensor& C, const Tensor& A, const Tensor& B) {
    int M = A.rows;
    int K = A.cols;
    int N = B.cols;
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A.at(i, k) * B.at(k, j);
            }
            C.at(i, j) = sum;
        }
    }
}