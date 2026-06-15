#pragma once
#include <cstdint>

#define QK8_0 32 // Bloco de 32 elementos para casar com AVX2 (256 bits)

struct BlockQ8_0 {
    float d;           // Delta: Fator de escala do bloco
    int8_t qs[QK8_0];  // 32 pesos quantizados em 8-bits
};

void quantize_row_q8_0(const float* x, BlockQ8_0* y, int k);