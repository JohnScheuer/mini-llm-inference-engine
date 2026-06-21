#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "cutlass_int8_gemm.h"

#define CUDA_CHECK(x) do { cudaError_t err=(x); if(err!=cudaSuccess){ \
std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; exit(1);} } while(0)

void run_cutlass_int8(
    int M, int N, int K,
    int8_t* d_A,
    int8_t* d_B,
    cutlass::half_t* d_C)
{
    CutlassInt8Gemm gemm_op;

    float alpha = 1.0f;
    float beta  = 0.0f;

    CutlassInt8Gemm::Arguments args(
        {M, N, K},
        {d_A, K},  // A é RowMajor (M x K), leading dim = K
        {d_B, K},  // B é ColMajor (K x N), leading dim = K
        {d_C, M},  // C é ColMajor (M x N), leading dim = M
        {d_C, M},
        {alpha, beta}
    );

    cutlass::Status status = gemm_op(args);

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM failed\n";
        exit(1);
    }
}

int main()
{
    int M = 512;
    int K = 512;
    int N = 8;

    int8_t* d_A;
    int8_t* d_B;
    cutlass::half_t* d_C;

    CUDA_CHECK(cudaMalloc(&d_A, M*K));
    CUDA_CHECK(cudaMalloc(&d_B, K*N));
    CUDA_CHECK(cudaMalloc(&d_C, M*N*sizeof(cutlass::half_t)));

    CUDA_CHECK(cudaMemset(d_A, 1, M*K));
    CUDA_CHECK(cudaMemset(d_B, 1, K*N));

    run_cutlass_int8(M, N, K, d_A, d_B, d_C);

    cudaDeviceSynchronize();

    std::cout << "CUTLASS INT8 GEMM ran successfully ✅\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}