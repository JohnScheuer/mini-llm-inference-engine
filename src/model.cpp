#include "model.h"
#include "layers.h"
#include <cstring>

void model_forward(Tensor& logits, Model& model, int token_id, int pos) {
    int dim = model.dim;
    
    // 1. Embedding
    Tensor x(1, dim);
    float* embed_row = &model.token_embedding.data[token_id * dim];
    std::memcpy(x.data.data(), embed_row, dim * sizeof(float));
    
    // 2. Camadas
    Tensor x_next(1, dim);
    for (int l = 0; l < model.n_layers; l++) {
        transformer_block_forward(x_next, x, model.layers[l], pos, dim, model.hidden_dim, model.n_heads);
        std::memcpy(x.data.data(), x_next.data.data(), dim * sizeof(float));
    }
    
    // 3. Norm final
    Tensor normed(1, dim);
    rmsnorm(normed, x, model.norm_final);
    
    // 4. Logits (Usa sua matmul_blocked!)
    matmul(logits, normed, model.lm_head);
}