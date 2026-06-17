#include <cuda_runtime.h>
#include <cstdio>
int main() {
    size_t size = 513 * 512; // kv_elements
    float *d;
    cudaError_t err = cudaMalloc(&d, size * sizeof(float));
    if (err != cudaSuccess) {
        printf("cudaMalloc falhou: %s\n", cudaGetErrorString(err));
    } else {
        printf("Alocado %zu bytes com sucesso\n", size * sizeof(float));
        cudaFree(d);
    }
    return 0;
}
