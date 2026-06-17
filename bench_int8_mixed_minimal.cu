#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do { cudaError_t e=call; if(e!=cudaSuccess) { fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1); } } while(0)
#define CUBLAS_CHECK(call) do { cublasStatus_t s=call; if(s!=CUBLAS_STATUS_SUCCESS) { fprintf(stderr,"cuBLAS %s:%d: %d\n",__FILE__,__LINE__,s); exit(1); } } while(0)

int main() {
    int M = 768, K = 768, N = 32;
    
    // Ambos INT8
    int8_t* h_W = (int8_t*)malloc(K * M * sizeof(int8_t));
    int8_t* h_x = (int8_t*)malloc(K * N * sizeof(int8_t));
    for (int i = 0; i < K * M; i++) h_W[i] = (int8_t)(rand() % 256 - 128);
    for (int i = 0; i < K * N; i++) h_x[i] = (int8_t)(rand() % 256 - 128);

    int8_t *d_W, *d_x;
    int32_t *d_y;  // ✅ INT32 (não float!)
    CUDA_CHECK(cudaMalloc(&d_W, K * M * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_x, K * N * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_y, M * N * sizeof(int32_t)));

    CUDA_CHECK(cudaMemcpy(d_W, h_W, K * M * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, K * N * sizeof(int8_t), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));  // ✅ Ativar Tensor Op

    int32_t alpha = 1, beta = 0;  // ✅ int32_t (não float)
    printf("Testando INT8 GEMM (M=%d, N=%d, K=%d)...\n", M, N, K);
    printf("  Inputs: CUDA_R_8I + CUDA_R_8I\n");
    printf("  Output: CUDA_R_32I (INT32 accumulation)\n");
    printf("  Compute: CUBLAS_COMPUTE_32I\n");
    printf("  Math Mode: CUBLAS_TENSOR_OP_MATH\n");
    
    cublasStatus_t stat = cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,  // ✅ OP_N para ambos (mais simples)
        M, N, K,
        &alpha,
        d_W, CUDA_R_8I, M,
        d_x, CUDA_R_8I, K,
        &beta,
        d_y, CUDA_R_32I, M,        // ✅ CUDA_R_32I (não 32F!)
        CUDA_R_32I,                 // ✅ Compute type = output type
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    if (stat == CUBLAS_STATUS_SUCCESS) {
        printf("✅ INT8 GEMM OK! Tensor Cores ativados.\n");
        
        // Verificar resultado
        int32_t* h_y = (int32_t*)malloc(M * N * sizeof(int32_t));
        CUDA_CHECK(cudaMemcpy(h_y, d_y, M * N * sizeof(int32_t), cudaMemcpyDeviceToHost));
        printf("   Primeiro elemento: %d (esperado: ~valor aleatório)\n", h_y[0]);
        free(h_y);
    } else {
        printf("❌ INT8 GEMM falhou: código %d\n", stat);
    }

    CUDA_CHECK(cudaFree(d_W)); CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y));
    CUBLAS_CHECK(cublasDestroy(handle));
    free(h_W); free(h_x);
    return 0;
}
