#pragma once
#include "tensor.h"
#include "layers.h"

struct TransformerLayer {
    Tensor wq, wk, wv, wo;
    Tensor w1, w2, w3;
    Tensor norm_attn;
    Tensor norm_ffn;
    KVCache cache;
    
    TransformerLayer(int dim, int hidden_dim, int max_seq)
        : wq(dim, dim), wk(dim, dim), wv(dim, dim), wo(dim, dim),
          w1(dim, hidden_dim), w2(hidden_dim, dim), w3(dim, hidden_dim),
          norm_attn(1, dim), norm_ffn(1, dim),
          cache(max_seq, dim) {}
};

void transformer_block_forward(
    Tensor& out,
    const Tensor& x,
    TransformerLayer& layer,
    int pos,
    int dim,
    int hidden_dim,
    int n_heads
);