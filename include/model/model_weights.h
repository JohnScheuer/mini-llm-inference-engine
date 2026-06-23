#pragma once

#include <vector>
#include <string>

namespace runtime {

// Representa pesos de uma camada Transformer
struct TransformerLayerWeights {

    // Attention
    std::vector<float> wq;
    std::vector<float> wk;
    std::vector<float> wv;
    std::vector<float> wo;

    // Feed-forward
    std::vector<float> w1;
    std::vector<float> w2;
    std::vector<float> w3;

    // Normalizações
    std::vector<float> norm_attn;
    std::vector<float> norm_ffn;
};

// Representa todos os pesos do modelo
struct ModelWeights {

    int dim = 0;
    int hidden_dim = 0;
    int n_layers = 0;
    int vocab_size = 0;

    std::vector<float> token_embedding;
    std::vector<TransformerLayerWeights> layers;
    std::vector<float> norm_final;
    std::vector<float> lm_head;
};

bool load_safetensors(const std::string& path, ModelWeights& weights);

} // namespace runtime