#pragma once
#include "model.h"
#include <string>
#include <cstdint>  // ← Esta linha resolve o problema

constexpr uint32_t MLLM_MAGIC = 0x4D4C4C4D;  // "MLLM" em ASCII
constexpr uint32_t MLLM_VERSION = 1;

struct ModelHeader {
    uint32_t magic;
    uint32_t version;
    int32_t vocab_size;
    int32_t dim;
    int32_t hidden_dim;
    int32_t n_layers;
    int32_t n_heads;
    int32_t max_seq_len;
    int32_t reserved[2];
};

bool save_model(const Model& model, const std::string& filepath);
Model* load_model(const std::string& filepath);