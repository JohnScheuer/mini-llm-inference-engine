#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t stat = call; \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, stat); \
        exit(1); \
    } \
} while(0)

int main() {
    const int M = 768, K = 768;
    const int nIter = 1000;

    // ✅ CORREÇÃO: sizeof() em todos os mallocs
    int8_t* h_W = (int8_t*)malloc(M * K * sizeof(int8_t));
    float* h_scales = (float*)malloc(M * sizeof(float));
    for (int i = 0; i < M * K; i++) h_W[i] = (int8_t)(rand() % 256 - 128);
    for (int i = 0; i < M; i++) h_scales[i] = (float)(rand() % 100) / 1000.0f;

    half* h_x = (half*)malloc(K * sizeof(half));
    for (int i = 0; i < K; i++) h_x[i] = __float2half((float)(rand() % 100) / 100.0f);

    int8_t *d_W;
    float *d_scales, *d_y;
    half *d_x;
    CUDA_CHECK(cudaMalloc(&d_W, M * K * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_scales, M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_y, M * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_W, h_W, M * K * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, h_scales, M * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, K * sizeof(half), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float alpha = 1.0f, beta = 0.0f;
    // Warmup
    CUBLAS_CHECK(cublasGemmEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        M, 1, K,
        &alpha,
        d_W, CUDA_R_8I, K,
        d_x, CUDA_R_16F, K,
        &beta,
        d_y, CUDA_R_32F, M,
        CUBLAS_COMPUTE_32F_FAST_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < nIter; i++) {
        CUBLAS_CHECK(cublasGemmEx(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            M, 1, K,
            &alpha,
            d_W, CUDA_R_8I, K,
            d_x, CUDA_R_16F, K,
            &beta,
            d_y, CUDA_R_32F, M,
            CUBLAS_COMPUTE_32F_FAST_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double avg_us = (ms * 1000.0) / nIter;
    printf("GEMM mista (%d×%d) PASSOU! Tempo médio: %.2f us\n", M, K, avg_us);

    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free(h_W); free(h_scales); free(h_x);
    return 0;
}
