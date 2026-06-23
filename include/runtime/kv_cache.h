#pragma once

#include <cstdint>

namespace runtime {

// Representa o KV cache de uma sequência (ou batch).
// O backend é responsável por alocar e gerenciar memória real.
struct KVCache {

    // Ponteiros device (opaque)
    void* k_cache = nullptr;
    void* v_cache = nullptr;

    // Estado
    int32_t current_seq_len = 0;
    int32_t max_seq_len = 0;

    // Dimensões
    int32_t head_dim = 0;
    int32_t n_heads = 0;
    int32_t n_layers = 0;

    // Batch (para decode_batch)
    int32_t batch_size = 1;

    // Para Tensor Parallel (dim local por shard)
    int32_t local_dim = 0;

    bool initialized() const {
        return k_cache != nullptr && v_cache != nullptr;
    }

    bool can_append(int32_t tokens = 1) const {
        return (current_seq_len + tokens) <= max_seq_len;
    }

    void reset() {
        current_seq_len = 0;
    }
};

} // namespace runtime