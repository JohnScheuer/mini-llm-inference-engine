#ifndef MATMUL_BLOCKED_H
#define MATMUL_BLOCKED_H

#ifdef __cplusplus
extern "C" {
#endif

// A interface limpa que o resto do seu engine vai usar
void matmul_blocked(int M, int N, int K, const float* A, const float* B, float* C);

#ifdef __cplusplus
}
#endif

#endif // MATMUL_BLOCKED_H