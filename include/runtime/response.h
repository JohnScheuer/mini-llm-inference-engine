#pragma once

#include <vector>
#include <cstdint>

namespace runtime {

// Representa a resposta de uma inferência.
// Pode conter apenas 1 token (decode)
// ou múltiplos tokens (prefill / geração completa).
struct InferenceResponse {

    // ID da requisição (eco do request)
    int64_t request_id = 0;

    // Tokens gerados
    std::vector<int32_t> output_tokens;

    // Logits opcionais (somente se solicitado)
    // Layout: [vocab_size]
    std::vector<float> logits;

    // Probabilidades opcionais (top-k)
    std::vector<float> probabilities;

    // Métricas (em microssegundos)
    uint64_t prefill_time_us = 0;
    uint64_t decode_time_us = 0;

    // Flags
    bool finished = false;   // true se geração terminou
    bool success = true;     // false se erro

    // Limpa conteúdo mantendo capacidade
    void clear() {
        output_tokens.clear();
        logits.clear();
        probabilities.clear();
        finished = false;
        success = true;
        prefill_time_us = 0;
        decode_time_us = 0;
    }

    bool has_logits() const {
        return !logits.empty();
    }

    bool has_probabilities() const {
        return !probabilities.empty();
    }
};

} // namespace runtime