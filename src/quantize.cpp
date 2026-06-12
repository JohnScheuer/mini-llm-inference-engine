#include "quantize.h"
#include <cstring>

// Conversão FP32 → FP16 (algoritmo manual, sem hardware)
// Implementação clássica do "half precision"
uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(float));
    
    uint32_t sign = (x >> 31) & 0x1;
    int32_t exponent = ((x >> 23) & 0xFF) - 127 + 15;  // ajusta bias
    uint32_t mantissa = x & 0x7FFFFF;
    
    uint16_t h;
    
    if (exponent <= 0) {
        // Underflow → zero
        h = (sign << 15);
    } else if (exponent >= 31) {
        // Overflow → infinito
        h = (sign << 15) | (0x1F << 10);
    } else {
        // Caso normal
        h = (sign << 15) | (exponent << 10) | (mantissa >> 13);
    }
    
    return h;
}

// Conversão FP16 → FP32
float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    
    uint32_t x;
    
    if (exponent == 0) {
        // Subnormal ou zero
        x = (sign << 31);
    } else if (exponent == 31) {
        // Infinito ou NaN
        x = (sign << 31) | (0xFF << 23) | (mantissa << 13);
    } else {
        // Caso normal
        uint32_t new_exp = exponent - 15 + 127;
        x = (sign << 31) | (new_exp << 23) | (mantissa << 13);
    }
    
    float f;
    std::memcpy(&f, &x, sizeof(float));
    return f;
}

// Tensor inteiro: FP32 → FP16
void tensor_to_fp16(const Tensor& src, std::vector<uint16_t>& dst) {
    dst.resize(src.data.size());
    for (size_t i = 0; i < src.data.size(); i++) {
        dst[i] = fp32_to_fp16(src.data[i]);
    }
}

// Tensor inteiro: FP16 → FP32
void fp16_to_tensor(const std::vector<uint16_t>& src, Tensor& dst) {
    for (size_t i = 0; i < src.size(); i++) {
        dst.data[i] = fp16_to_fp32(src[i]);
    }
}