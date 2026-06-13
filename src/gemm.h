#pragma once

void gemm(
    int M, int N, int K,
    const float* A,
    const float* B,
    float* C
);