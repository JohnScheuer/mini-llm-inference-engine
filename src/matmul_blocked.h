#pragma once
#include "tensor_int8.h"

// Motor GEMV de alta performance
void matmul_gemv_int8_avx2(int N, int K, const BlockQ8_0* W, const BlockQ8_0* X, float* Y);

// Wrapper para inferência
void matmul_blocked_int8(int M, int N, int K, const BlockQ8_0* W, const float* X_fp32, float* Y);

// Fallback para compatibilidade
void matmul_blocked(int M, int N, int K, const float* A, const float* B, float* C);