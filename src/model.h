#pragma once
#include "tensor.h"
#include "layers.h"
#include "transformer.h"
#include <vector>

struct Model {
    int vocab_size;
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int max_seq_len;
    
    Tensor token_embedding;
    std::vector<TransformerLayer> layers;
    Tensor norm_final;
    Tensor lm_head;
    
    Model(int vocab_size, int dim, int hidden_dim, 
          int n_layers, int n_heads, int max_seq_len)
        : vocab_size(vocab_size),
          dim(dim),
          hidden_dim(hidden_dim),
          n_layers(n_layers),
          n_heads(n_heads),
          max_seq_len(max_seq_len),
          token_embedding(vocab_size, dim),
          norm_final(1, dim),
          lm_head(dim, vocab_size)
    {
        for (int i = 0; i < n_layers; i++) {
            layers.emplace_back(dim, hidden_dim, max_seq_len);
        }
    }
};

void model_forward(Tensor& logits, Model& model, int token_id, int pos);

// NOVO: Função que quantizará os pesos logo após carregar do disco
void quantize_model_weights(Model& model);