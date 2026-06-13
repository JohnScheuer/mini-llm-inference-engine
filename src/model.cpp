#include "model.h"
#include "layers.h"
#include "transformer.h"
#include "matmul_blocked.h"
#include <cstring>

void model_forward(Tensor& logits, Model& model, int token_id, int pos) {
    int dim = model.dim;
    // Garante que o tensor de logits tenha o tamanho do vocabulário
    if (logits.data.size() < (size_t)model.vocab_size) logits.data.resize(model.vocab_size);

    // 1. Embedding
    Tensor x(1, dim);
    std::memcpy(x.data.data(), &model.token_embedding.data[token_id * dim], dim * sizeof(float));

    // 2. Blocos Transformer
    Tensor x_next(1, dim);
    for (int l = 0; l < model.n_layers; l++) {
        transformer_block_forward(x_next, x, model.layers[l], pos, dim, model.hidden_dim, model.n_heads);
        std::memcpy(x.data.data(), x_next.data.data(), dim * sizeof(float));
    }

    // 3. RMSNorm final
    Tensor normed(1, dim);
    rmsnorm(normed, x, model.norm_final);

    // 4. LM Head (Projeção para o vocabulário)
    // Aqui usamos o atalho M=1 otimizado
    matmul_blocked(1, model.vocab_size, dim, normed.data.data(), model.lm_head.data.data(), logits.data.data());
}