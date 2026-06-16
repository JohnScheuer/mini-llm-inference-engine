// Truque para esconder as funções conflitantes do glibc antes de incluir o CUDA
#define cospi __hidden_cospi
#define sinpi __hidden_sinpi
#define rsqrt __hidden_rsqrt
#define cospif __hidden_cospif
#define sinpif __hidden_sinpif
#define rsqrtf __hidden_rsqrtf

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Restaura os nomes normais para o nosso código usar as versões rápidas do CUDA
#undef cospi
#undef sinpi
#undef rsqrt
#undef cospif
#undef sinpif
#undef rsqrtf

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

__global__ void rmsnorm_kernel(const float* __restrict__ in,
                               const float* __restrict__ gamma,
                               float* __restrict__ out,
                               int dim, float eps) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    const float* x = in + blockIdx.x * dim;
    float* o = out + blockIdx.x * dim;
    
    float local = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x)
        local += x[i] * x[i];
    sdata[tid] = local;
    __syncthreads();
    
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    float scale = rsqrtf(sdata[0] / dim + eps);
    for (int i = tid; i < dim; i += blockDim.x)
        o[i] = x[i] * scale * gamma[i];
}

__global__ void rope_kernel(float* __restrict__ q, float* __restrict__ k,
                            const float* __restrict__ cos,
                            const float* __restrict__ sin,
                            int pos, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = dim / 2;
    if (idx >= half_dim) return;
    int i = idx * 2;
    float c = cos[pos * dim + i];
    float s = sin[pos * dim + i];
    float v0_q = q[i], v1_q = q[i+1];
    q[i] = v0_q * c - v1_q * s; q[i+1] = v0_q * s + v1_q * c;
    float v0_k = k[i], v1_k = k[i+1];
    k[i] = v0_k * c - v1_k * s; k[i+1] = v0_k * s + v1_k * c;
}

__global__ void silu_mul_kernel(float* __restrict__ g, const float* __restrict__ u, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    float x = g[i];
    g[i] = (x / (1.0f + expf(-x))) * u[i];
}

float* cuda_alloc_float(int n) {
    float* d; CUDA_CHECK(cudaMalloc(&d, n * sizeof(float))); return d;
}

void cuda_rand(float* d, int n) {
    float* h = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) h[i] = (float)(rand() % 100) / 100.0f;
    CUDA_CHECK(cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice));
    free(h);
}

int main() {
    int device; CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("========================================\n");
    printf(" Mini-LLM CUDA Kernel Benchmark\n");
    printf("========================================\n");
    printf(" GPU: %s\n", prop.name);
    printf(" CC:  %d.%d\n", prop.major, prop.minor);
    printf(" SMs: %d\n", prop.multiProcessorCount);
    printf(" Mem: %lu MB\n", prop.totalGlobalMem >> 20);
    printf("========================================\n\n");
    
    int dim = 288, hidden_dim = 768, n_layers = 6, n_iter = 10000;
    printf("Config: dim=%d hidden=%d layers=%d\n\n", dim, hidden_dim, n_layers);
    
    float *d_in = cuda_alloc_float(dim), *d_gamma = cuda_alloc_float(dim), *d_out = cuda_alloc_float(dim);
    float *d_q = cuda_alloc_float(dim), *d_k = cuda_alloc_float(dim), *d_cos = cuda_alloc_float(dim), *d_sin = cuda_alloc_float(dim);
    float *d_g = cuda_alloc_float(hidden_dim), *d_u = cuda_alloc_float(hidden_dim);
    cuda_rand(d_in, dim); cuda_rand(d_gamma, dim); cuda_rand(d_q, dim); cuda_rand(d_k, dim);
    cuda_rand(d_cos, dim); cuda_rand(d_sin, dim); cuda_rand(d_g, hidden_dim); cuda_rand(d_u, hidden_dim);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < n_iter; i++) rmsnorm_kernel<<<1, 256, 256*sizeof(float)>>>(d_in, d_gamma, d_out, dim, 1e-5f);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms_rms = 0; cudaEventElapsedTime(&ms_rms, start, stop); ms_rms /= n_iter;
    
    cudaEventRecord(start);
    for (int i = 0; i < n_iter; i++) rope_kernel<<<(dim/2+255)/256, 256>>>(d_q, d_k, d_cos, d_sin, 128, dim);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms_rope = 0; cudaEventElapsedTime(&ms_rope, start, stop); ms_rope /= n_iter;
    
    cudaEventRecord(start);
    for (int i = 0; i < n_iter; i++) silu_mul_kernel<<<(hidden_dim+255)/256, 256>>>(d_g, d_u, hidden_dim);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms_silu = 0; cudaEventElapsedTime(&ms_silu, start, stop); ms_silu /= n_iter;
    
    printf("RMSNorm: %.1f us\n", ms_rms * 1000.0f);
    printf("RoPE:    %.1f us\n", ms_rope * 1000.0f);
    printf("SiLU:    %.1f us\n", ms_silu * 1000.0f);
    
    float total = (ms_rms + ms_rope + ms_silu) * n_layers;
    printf("\n========================================\n");
    printf("Total (6 layers): %.4f ms\n", total);
    printf("Est. throughput:  %.0f tok/s (kernels only)\n", 1000.0f/total);
    printf("========================================\n");
    
    return 0;
}
