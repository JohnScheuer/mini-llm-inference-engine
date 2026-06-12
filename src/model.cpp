#include "model.h"
#include <cstring>

void model_forward(
    Tensor& logits,
    Model& model,
    int token_id,
    int pos
) {
    int dim = model.dim;
    
    // 1. EMBEDDING: pega a linha do token_id da tabela de embedding
    Tensor x(1, dim);
    for (int i = 0; i < dim; i++) {
        x.data[i] = model.token_embedding.at(token_id, i);
    }
    
    // 2. Passa por todas as N camadas do Transformer
    Tensor temp(1, dim);
    for (int l = 0; l < model.n_layers; l++) {
        transformer_block_forward(
            temp, x, model.layers[l], 
            pos, dim, model.hidden_dim, model.n_heads
        );
        std::memcpy(x.data.data(), temp.data.data(), dim * sizeof(float));
    }
    
    // 3. RMSNorm final
    Tensor normed(1, dim);
    rmsnorm(normed, x, model.norm_final);
    
    // 4. LM Head: projeta para vocab_size logits
    matmul(logits, normed, model.lm_head);
}