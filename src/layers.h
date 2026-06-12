#pragma once
#include "tensor.h"
#include <vector>

// RMSNorm
void rmsnorm(Tensor& out, const Tensor& x, const Tensor& gamma, float eps = 1e-5f);

// RoPE
void apply_rope(Tensor& x, int pos, int dim);

// MatMul
void matmul(Tensor& out, const Tensor& a, const Tensor& b);

// Softmax
void softmax(float* x, int size);

// KV-Cache
struct KVCache {
    std::vector<float> k_cache;
    std::vector<float> v_cache;
    int max_seq_len;
    int dim;

    KVCache(int max_seq, int d) 
        : k_cache(max_seq * d, 0.0f), 
          v_cache(max_seq * d, 0.0f), 
          max_seq_len(max_seq), 
          dim(d) {}
};

// Multi-Head Attention com KV-Cache
void attention_forward(
    Tensor& out,
    const Tensor& x,
    const Tensor& wq,
    const Tensor& wk,
    const Tensor& wv,
    const Tensor& wo,
    KVCache& cache,
    int pos,
    int dim,
    int n_heads
);

// SiLU
float silu(float x);

// FFN com SwiGLU
void ffn_forward(
    Tensor& out,
    const Tensor& x,
    const Tensor& w1,
    const Tensor& w2,
    const Tensor& w3,
    int dim,
    int hidden_dim
);
// MatMul genérica: C[M x N] = A[M x K] * B[K x N]
// Versão ingênua (vai ser o baseline pra comparar)
void matmul_naive(Tensor& C, const Tensor& A, const Tensor& B);