#pragma once

#include <vector>
#include <cstdint>

namespace runtime {

// ============================================================
// Tipo da requisição
// ============================================================

enum class RequestType {
    PREFILL,   // processar prompt completo
    DECODE,    // gerar 1 token (hot path)
    GENERATE   // prefill + decode até max_new_tokens
};

// ============================================================
// Parâmetros de amostragem
// ============================================================

struct SamplingParams {
    float temperature = 1.0f;
    int32_t top_k = 0;              // 0 = desativado
    float top_p = 1.0f;             // 1.0 = desativado
    float repetition_penalty = 1.0f;
    uint64_t seed = 0;
};

// ============================================================
// Requisição de inferência
// ============================================================

struct InferenceRequest {

    // Identificação (útil para serving assíncrono)
    int64_t request_id = 0;

    // Tipo da requisição
    RequestType type = RequestType::GENERATE;

    // Tokens de entrada
    std::vector<int32_t> input_tokens;

    // Tokens de parada opcionais
    std::vector<int32_t> stop_tokens;

    // Número máximo de tokens novos a gerar
    int32_t max_new_tokens = 1;

    // Parâmetros de geração
    SamplingParams sampling;

    // Se true, backend deve retornar logits completos
    bool return_logits = false;

    // ========================================================
    // Validação robusta
    // ========================================================
    bool is_valid() const {

        // Tokens sempre necessários
        if (input_tokens.empty())
            return false;

        switch (type) {

            case RequestType::PREFILL:
                // Apenas precisa de tokens válidos
                return true;

            case RequestType::DECODE:
                // Decode de 1 token deve receber exatamente 1 token
                if (input_tokens.size() != 1)
                    return false;
                return true;

            case RequestType::GENERATE:
                // Precisa de tokens e max_new_tokens > 0
                if (max_new_tokens <= 0)
                    return false;
                return true;

            default:
                return false;
        }
    }
};

} // namespace runtime