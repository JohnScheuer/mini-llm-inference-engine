#pragma once
#include "model.h"
#include <vector>

// Sampling: escolhe o próximo token a partir dos logits
int sample_argmax(const Tensor& logits);
int sample_temperature(const Tensor& logits, float temperature);
int sample_top_k(const Tensor& logits, float temperature, int top_k);

// Geração: dado um prompt, gera N tokens
std::vector<int> generate(
    Model& model,
    const std::vector<int>& prompt,
    int max_new_tokens,
    float temperature = 1.0f,
    int top_k = 0  // 0 = sem top-k
);