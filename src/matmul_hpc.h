#pragma once

constexpr int BM = 64, BN = 256, BK = 64;

void matmul_naive(float* C, const float* A, const float* B, int M, int N, int K);
void matmul_ikj(float* C, const float* A, const float* B, int M, int N, int K);
void matmul_tiled(float* C, const float* A, const float* B, int M, int N, int K);
void matmul_simd(float* C, const float* A, const float* B, int M, int N, int K);
void benchmark_matmul(int M, int N, int K);
