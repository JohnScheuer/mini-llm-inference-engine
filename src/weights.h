#pragma once
#include "model.h"
#include <string>
#include <cstdint>

constexpr uint32_t MLLM_MAGIC = 0x4D4C4C4D;
constexpr uint32_t MLLM_VERSION = 1;

// Tipo de quantização
enum QuantType : uint32_t {
    QUANT_FP32 = 0,
    QUANT_FP16 = 1
};

struct ModelHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t quant_type;  // ← NOVO: tipo de quantização
    int32_t vocab_size;
    int32_t dim;
    int32_t hidden_dim;
    int32_t n_layers;
    int32_t n_heads;
    int32_t max_seq_len;
    int32_t reserved;  // padding
};

// Salva o modelo, com opção de quantização
bool save_model(const Model& model, const std::string& filepath, 
                QuantType quant = QUANT_FP32);

// Carrega o modelo (detecta a quantização automaticamente)
Model* load_model(const std::string& filepath);