#pragma once

#include <cstdint>
#include <string>

namespace runtime {

// Configuração estrutural do modelo.
// Não contém pesos — apenas dimensões e metadados.
struct ModelConfig {

    // Arquitetura principal
    int32_t dim = 0;
    int32_t hidden_dim = 0;
    int32_t n_layers = 0;
    int32_t n_heads = 0;
    int32_t n_kv_heads = 0;
    int32_t head_dim = 0;
    int32_t vocab_size = 0;
    int32_t max_seq_len = 0;

    // Tipo de precisão
    enum class Precision {
        FP16,
        INT8,
        MIXED_INT8_FP16,
        BF16,
        FP8
    };

    Precision precision = Precision::FP16;

    // Runtime flags
    bool use_flash_attention = false;
    bool use_tensor_parallel = false;
    bool use_fused_mlp = false;

    // Caminho do modelo
    std::string model_path;

    // Validação estrutural
    bool is_valid() const {
        if (dim <= 0 ||
            hidden_dim <= 0 ||
            n_layers <= 0 ||
            n_heads <= 0 ||
            vocab_size <= 0 ||
            max_seq_len <= 0)
            return false;

        if (dim % n_heads != 0)
            return false;

        if (n_kv_heads > n_heads)
            return false;

        return true;
    }

    // Derivados úteis
    int32_t computed_head_dim() const {
        return dim / n_heads;
    }
};

} // namespace runtime