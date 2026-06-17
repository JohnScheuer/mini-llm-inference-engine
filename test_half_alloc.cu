#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
int main() {
    const int K = 768;
    half* h_x = (half*)malloc(K * sizeof(half));
    for (int i = 0; i < K; i++) h_x[i] = __float2half(1.0f);
    half* d_x;
    cudaMalloc(&d_x, K * sizeof(half));
    cudaMemcpy(d_x, h_x, K * sizeof(half), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaFree(d_x);
    free(h_x);
    printf("Alocação de half OK\n");
    return 0;
}
