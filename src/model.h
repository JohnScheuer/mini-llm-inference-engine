// model.h
#ifndef MODEL_H
#define MODEL_H

#include "tensor_int8.h"
#include <vector>
#include <string>
#include <cmath>

struct KVCache {
    std::vector<float> k_cache;
    std::vector<float> v_cache;

    KVCache() {}
    KVCache(int max_seq, int dim)
        : k_cache(max_seq * dim, 0.0f),
          v_cache(max_seq * dim, 0.0f) {}
};

struct TransformerLayer {
    Tensor wq, wk, wv, wo;
    Tensor w1, w2, w3;
    Tensor norm_attn;
    Tensor norm_ffn;

    TransformerLayer() {}
    TransformerLayer(int dim, int hidden_dim)
        : wq(dim, dim), wk(dim, dim), wv(dim, dim), wo(dim, dim),
          w1(dim, hidden_dim), w2(hidden_dim, dim), w3(dim, hidden_dim),
          norm_attn(1, dim), norm_ffn(1, dim) {}
};

struct Model {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int vocab_size;
    int max_seq_len;

    Tensor token_embedding;
    std::vector<TransformerLayer> layers;
    Tensor norm_final;
    Tensor lm_head;
};

bool load_model_weights(Model& model, const std::string& path);

#endif