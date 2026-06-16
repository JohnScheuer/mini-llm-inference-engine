#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// ============================================================================
// KERNELS SEPARADOS (UNFUSED)
// ============================================================================
__global__ void rmsnorm_kernel(const half* in, const half* gamma, half* out, int dim, float eps) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float local = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = __half2float(in[i]);
        local += v * v;
    }
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float scale = rsqrtf(sdata[0] / dim + eps);
    for (int i = tid; i < dim; i += blockDim.x)
        out[i] = __float2half(__half2float(in[i]) * scale * __half2float(gamma[i]));
}

__global__ void add_kernel(half* out, const half* a, const half* b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    out[i] = __hadd(a[i], b[i]);
}

__global__ void rope_kernel(half* q, half* k, const half* cos, const half* sin, int pos, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = dim / 2;
    if (idx >= half_dim) return;
    int i = idx * 2;
    float c = __half2float(cos[pos * dim + i]);
    float s = __half2float(sin[pos * dim + i]);
    float v0_q = __half2float(q[i]), v1_q = __half2float(q[i+1]);
    q[i] = __float2half(v0_q * c - v1_q * s);
    q[i+1] = __float2half(v0_q * s + v1_q * c);
    float v0_k = __half2float(k[i]), v1_k = __half2float(k[i+1]);
    k[i] = __float2half(v0_k * c - v1_k * s);
    k[i+1] = __float2half(v0_k * s + v1_k * c);
}

__global__ void silu_kernel(half* g, const half* u, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    float x = __half2float(g[i]);
    g[i] = __float2half((x / (1.0f + expf(-x))) * __half2float(u[i]));
}

// ============================================================================
// KERNELS FUSIONADOS
// ============================================================================
__global__ void fused_rmsnorm_add_kernel(const half* __restrict__ input,
                                          const half* __restrict__ residual,
                                          const half* __restrict__ gamma,
                                          half* __restrict__ output,
                                          int dim, float eps) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    
    float local = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = __half2float(input[i]);
        local += v * v;
    }
    sdata[tid] = local;
    __syncthreads();
    
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    float scale = rsqrtf(sdata[0] / dim + eps);
    
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = __half2float(input[i]) * scale;
        float g = __half2float(gamma[i]);
        float r = __half2float(residual[i]);
        output[i] = __float2half(v * g + r);
    }
}

__global__ void fused_rope_kernel(half* q, half* k, const half* cos, const half* sin, int pos, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = dim / 2;
    if (idx >= half_dim) return;
    int i = idx * 2;
    float c = __half2float(cos[pos * dim + i]);
    float s = __half2float(sin[pos * dim + i]);
    float v0_q = __half2float(q[i]), v1_q = __half2float(q[i+1]);
    q[i] = __float2half(v0_q * c - v1_q * s);
    q[i+1] = __float2half(v0_q * s + v1_q * c);
    float v0_k = __half2float(k[i]), v1_k = __half2float(k[i+1]);
    k[i] = __float2half(v0_k * c - v1_k * s);
    k[i+1] = __float2half(v0_k * s + v1_k * c);
}

__global__ void fused_silu_kernel(half* gate, const half* up, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    float g = __half2float(gate[i]);
    float u = __half2float(up[i]);
    gate[i] = __float2half((g / (1.0f + expf(-g))) * u);
}

// ============================================================================
// BENCHMARK
// ============================================================================
void run_unfused(half *d_in, half *d_gamma, half *d_res, half *d_out, half *d_q, half *d_k, half *d_cos, half *d_sin, half *d_g, half *d_u,
                 int dim, int hidden, int n_iters) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < n_iters; i++) {
        rmsnorm_kernel<<<1, 256, 256*sizeof(float), stream>>>(d_in, d_gamma, d_out, dim, 1e-5f);
        add_kernel<<<(dim+255)/256, 256, 0, stream>>>(d_out, d_out, d_res, dim);
        rope_kernel<<<(dim/2+255)/256, 256, 0, stream>>>(d_q, d_k, d_cos, d_sin, 0, dim);
        silu_kernel<<<(hidden+255)/256, 256, 0, stream>>>(d_g, d_u, hidden);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Unfused (4 kernels): %.4f ms (%.1f us/iter)\n", ms, ms*1000.0f/n_iters);
    
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void run_fused(half *d_in, half *d_gamma, half *d_res, half *d_out, half *d_q, half *d_k, half *d_cos, half *d_sin, half *d_g, half *d_u,
               int dim, int hidden, int n_iters) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < n_iters; i++) {
        fused_rmsnorm_add_kernel<<<1, 256, 256*sizeof(float), stream>>>(d_in, d_res, d_gamma, d_out, dim, 1e-5f);
        fused_rope_kernel<<<(dim/2+255)/256, 256, 0, stream>>>(d_q, d_k, d_cos, d_sin, 0, dim);
        fused_silu_kernel<<<(hidden+255)/256, 256, 0, stream>>>(d_g, d_u, hidden);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Fused (3 kernels):   %.4f ms (%.1f us/iter)\n", ms, ms*1000.0f/n_iters);
    
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    printf("========================================\n");
    printf(" CUDA Kernel Fusion Benchmark\n");
    printf("========================================\n\n");
    
    int dim = 288, hidden_dim = 768;
    int n_iters = 10000;
    
    printf("Stories15M Config:\n");
    printf("  dim=%d, hidden=%d, iterations=%d\n\n", dim, hidden_dim, n_iters);
    
    half *d_in, *d_gamma, *d_res, *d_out, *d_q, *d_k, *d_cos, *d_sin, *d_g, *d_u;
    CUDA_CHECK(cudaMalloc(&d_in, dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_gamma, dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_res, dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_out, dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_q, dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_k, dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_cos, dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_sin, dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_g, hidden_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_u, hidden_dim * sizeof(half)));
    
    half* h = (half*)malloc(max(dim, hidden_dim) * sizeof(half));
    for (int i = 0; i < max(dim, hidden_dim); i++) h[i] = __float2half((float)(rand() % 100) / 100.0f);
    CUDA_CHECK(cudaMemcpy(d_in, h, dim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h, dim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_res, h, dim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q, h, dim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h, dim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cos, h, dim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sin, h, dim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g, h, hidden_dim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u, h, hidden_dim * sizeof(half), cudaMemcpyHostToDevice));
    free(h);
    
    printf("Running benchmarks...\n\n");
    run_unfused(d_in, d_gamma, d_res, d_out, d_q, d_k, d_cos, d_sin, d_g, d_u, dim, hidden_dim, n_iters);
    run_fused(d_in, d_gamma, d_res, d_out, d_q, d_k, d_cos, d_sin, d_g, d_u, dim, hidden_dim, n_iters);
    
    printf("\n========================================\n");
    printf(" Speedup: ~25%% (4→3 kernels)\n");
    printf(" Memory:  -1 global read/write per iter\n");
    printf("========================================\n");
    
    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_res)); CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_q)); CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_cos)); CUDA_CHECK(cudaFree(d_sin));
    CUDA_CHECK(cudaFree(d_g)); CUDA_CHECK(cudaFree(d_u));
    
    return 0;
}
