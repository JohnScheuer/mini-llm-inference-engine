#ifndef MODEL_H
#define MODEL_H
#include <vector>

struct Tensor {
    std::vector<float> data;
    int r, c;
    Tensor() : r(0), c(0) {}
    Tensor(int rows, int cols) : r(rows), c(cols) { data.resize(rows * cols); }
};

struct Layer {
    Tensor wq, wk, wv, wo;
    Tensor w1, w2, w3;
    Tensor norm_attn, norm_ffn;
};

struct Model {
    int dim, hidden_dim, n_layers, n_heads, vocab_size, max_seq_len;
    Tensor token_embedding, norm_final, lm_head;
    std::vector<Layer> layers;
};
#endif
