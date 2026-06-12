#pragma once
#include "tensor.h"
#include <cstdint>

// Converte FP32 (float) para FP16 (representado como uint16_t)
uint16_t fp32_to_fp16(float f);

// Converte FP16 (uint16_t) de volta para FP32 (float)
float fp16_to_fp32(uint16_t h);

// Converte um Tensor inteiro de FP32 para um vetor de FP16
void tensor_to_fp16(const Tensor& src, std::vector<uint16_t>& dst);

// Converte um vetor de FP16 para um Tensor FP32
void fp16_to_tensor(const std::vector<uint16_t>& src, Tensor& dst);