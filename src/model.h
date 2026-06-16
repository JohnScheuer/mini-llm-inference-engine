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

struct RunState {
    std::vector<float> x;
    std::vector<float> xb;
    std::vector<float> q, k, v;
    std::vector<float> attn_out;
    std::vector<float> ffn_g;
    std::vector<float> ffn_u;
    std::vector<float> ffn_out;
    std::vector<float> logits;
    std::vector<float> scores;
    std::vector<BlockQ8_0> xq;
    std::vector<float> rope_cos;
    std::vector<float> rope_sin;
};

inline void init_run_state(RunState& s, const Model& m) {
    int dim = m.dim;
    int hidden_dim = m.hidden_dim;

    s.x.resize(dim);
    s.xb.resize(dim);
    s.q.resize(dim);
    s.k.resize(dim);
    s.v.resize(dim);
    s.attn_out.resize(dim);
    s.ffn_g.resize(hidden_dim);
    s.ffn_u.resize(hidden_dim);
    s.ffn_out.resize(dim);
    s.logits.resize(m.vocab_size);
    s.scores.resize(m.max_seq_len * m.n_heads);

    int max_dim = (dim > hidden_dim) ? dim : hidden_dim;
    s.xq.resize(max_dim / QK8_0 + 1);

    s.rope_cos.resize(m.max_seq_len * dim);
    s.rope_sin.resize(m.max_seq_len * dim);
    for (int pos = 0; pos < m.max_seq_len; pos++) {
        for (int i = 0; i < dim; i += 2) {
            float freq = 1.0f / powf(10000.0f, (float)i / dim);
            float val = (float)pos * freq;
            float cv = cosf(val);
            float sv = sinf(val);
            s.rope_cos[pos * dim + i]     = cv;
            s.rope_sin[pos * dim + i]     = sv;
            s.rope_cos[pos * dim + i + 1] = cv;
            s.rope_sin[pos * dim + i + 1] = sv;
        }
    }
}

bool load_model_weights(Model& model, const std::string& path);

#endif