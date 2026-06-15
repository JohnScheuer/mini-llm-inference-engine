#include "model.h"
#include "layers.h"
#include "transformer.h"
#include "matmul_blocked.h"
#include <cstdio>
#include <cstring>

void model_forward(Tensor& logits, Model& model, int token_id, int pos) {
    int dim = model.dim;
    Tensor x(1, dim);
    std::memcpy(x.data.data(), &model.token_embedding.data[token_id * dim], dim * sizeof(float));

    Tensor x_next(1, dim);
    for (int l = 0; l < model.n_layers; l++) {
        transformer_block_forward(x_next, x, model.layers[l], pos, dim, model.hidden_dim, model.n_heads);
        std::memcpy(x.data.data(), x_next.data.data(), dim * sizeof(float));
    }
    Tensor normed(1, dim);
    rmsnorm(normed, x, model.norm_final);
    matmul_blocked_int8(1, model.vocab_size, dim, model.lm_head.q_data.data(), normed.data.data(), logits.data.data());
}

bool load_model_weights(Model& model, const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    int h[7];
    if (fread(h, sizeof(int), 7, f) != 7) { fclose(f); return false; }
    model.dim = h[0]; model.hidden_dim = h[1]; model.n_layers = h[2]; model.n_heads = h[3];
    model.vocab_size = h[5]; model.max_seq_len = h[6];
    model.token_embedding = Tensor(model.vocab_size, model.dim);
    model.layers.clear();
    for(int i=0; i<model.n_layers; i++) model.layers.emplace_back(model.dim, model.hidden_dim, model.max_seq_len);
    model.norm_final = Tensor(1, model.dim);
    model.lm_head = Tensor(model.vocab_size, model.dim);
    auto read_f = [&](float* p, size_t n) { return fread(p, sizeof(float), n, f) == n; };
    read_f(model.token_embedding.data.data(), (size_t)model.vocab_size * model.dim);
    for (int i = 0; i < model.n_layers; i++) read_f(model.layers[i].norm_attn.data.data(), model.dim);
    for (int i = 0; i < model.n_layers; i++) read_f(model.layers[i].wq.data.data(), (size_t)model.dim * model.dim);
    for (int i = 0; i < model.n_layers; i++) read_f(model.layers[i].wk.data.data(), (size_t)model.dim * model.dim);
    for (int i = 0; i < model.n_layers; i++) read_f(model.layers[i].wv.data.data(), (size_t)model.dim * model.dim);
    for (int i = 0; i < model.n_layers; i++) read_f(model.layers[i].wo.data.data(), (size_t)model.dim * model.dim);
    for (int i = 0; i < model.n_layers; i++) read_f(model.layers[i].norm_ffn.data.data(), model.dim);
    for (int i = 0; i < model.n_layers; i++) read_f(model.layers[i].w1.data.data(), (size_t)model.dim * model.hidden_dim);
    for (int i = 0; i < model.n_layers; i++) read_f(model.layers[i].w2.data.data(), (size_t)model.hidden_dim * model.dim);
    for (int i = 0; i < model.n_layers; i++) read_f(model.layers[i].w3.data.data(), (size_t)model.dim * model.hidden_dim);
    read_f(model.norm_final.data.data(), model.dim);
    read_f(model.lm_head.data.data(), (size_t)model.vocab_size * model.dim);
    fclose(f);
    return true;
}