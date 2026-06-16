#include <cuda_runtime.h>
#include <cublas_v2.h>
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

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t stat = call; \
        if (stat != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, stat); \
            exit(1); \
        } \
    } while(0)

// ============================================================================
// FUSED KERNELS
// ============================================================================

// FUSÃO 1: RMSNorm + Add Residual em um único kernel
// Economiza: 1 kernel launch + 1 leitura/escrita global memory
__global__ void fused_rmsnorm_add_kernel(const half* __restrict__ input,
                                          const half* __restrict__ residual,
                                          const half* __restrict__ gamma,
                                          half* __restrict__ output,
                                          int dim, float eps) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    
    // Pass 1: Calcular soma dos quadrados (RMSNorm)
    float local = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = __half2float(input[i]);
        local += v * v;
    }
    sdata[tid] = local;
    __syncthreads();
    
    // Redução na shared memory
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    float scale = rsqrtf(sdata[0] / dim + eps);
    
    // Pass 2: Normalizar + Gamma + Add Residual (TUDO EM UM KERNEL)
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = __half2float(input[i]) * scale;
        float g = __half2float(gamma[i]);
        float r = __half2float(residual[i]);
        output[i] = __float2half(v * g + r);  // FUSÃO: Norm + Gamma + Residual
    }
}

// FUSÃO 2: RoPE aplicado diretamente após QKV (evita kernel separado)
__global__ void fused_rope_qkv_kernel(half* __restrict__ q,
                                       half* __restrict__ k,
                                       const half* __restrict__ cos,
                                       const half* __restrict__ sin,
                                       int pos, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = dim / 2;
    if (idx >= half_dim) return;
    
    int i = idx * 2;
    float c = __half2float(cos[pos * dim + i]);
    float s = __half2float(sin[pos * dim + i]);
    
    // Q RoPE
    float v0_q = __half2float(q[i]), v1_q = __half2float(q[i+1]);
    q[i] = __float2half(v0_q * c - v1_q * s);
    q[i+1] = __float2half(v0_q * s + v1_q * c);
    
    // K RoPE
    float v0_k = __half2float(k[i]), v1_k = __half2float(k[i+1]);
    k[i] = __float2half(v0_k * c - v1_k * s);
    k[i+1] = __float2half(v0_k * s + v1_k * c);
}

// FUSÃO 3: SiLU + Multiplication (já feito, mas otimizado)
__global__ void fused_silu_gate_kernel(half* __restrict__ gate,
                                        const half* __restrict__ up,
                                        int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    float g = __half2float(gate[i]);
    float u = __half2float(up[i]);
    gate[i] = __float2half((g / (1.0f + expf(-g))) * u);
}

// ============================================================================
// BENCHMARK COMPARATIVO
// ============================================================================

void benchmark_unfused(half *d_input, half *d_residual, half *d_gamma, 
                       half *d_output, half *d_q, half *d_k,
                       half *d_cos, half *d_sin, half *d_gate, half *d_up,
                       int dim, int hidden_dim, int n_iters) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Kernels separados (unfused)
    cudaEventRecord(start);
    for (int i = 0; i < n_iters; i++) {
        // RMSNorm (kernel 1)
        extern __shared__ float sdata[];
        // ... (kernel RMSNorm separado)
        
        // Add Residual (kernel 2)
        // ... (kernel Add separado)
        
        // RoPE (kernel 3)
        // ... (kernel RoPE separado)
        
        // SiLU (kernel 4)
        // ... (kernel SiLU separado)
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Unfused (4 kernels): %.3f ms\n", ms / n_iters);
    
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void benchmark_fused(half *d_input, half *d_residual, half *d_gamma,
                     half *d_output, half *d_q, half *d_k,
                     half *d_cos, half *d_sin, half *d_gate, half *d_up,
                     int dim, int hidden_dim, int n_iters) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Kernels fusionados
    cudaEventRecord(start);
    for (int i = 0; i < n_iters; i++) {
        // Fused RMSNorm + Add (1 kernel ao invés de 2)
        fused_rmsnorm_add_kernel<<<1, 256, 256*sizeof(float), stream>>>(
            d_input, d_residual, d_gamma, d_output, dim, 1e-5f);
        
        // Fused RoPE (otimizado)
        fused_rope_qkv_kernel<<<(dim/2+255)/256, 256, 0, stream>>>(
            d_q, d_k, d_cos, d_sin, 0, dim);
        
        // Fused SiLU
        fused_silu_gate_kernel<<<(hidden_dim+255)/256, 256, 0, stream>>>(
            d_gate, d_up, hidden_dim);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Fused (3 kernels):  %.3f ms\n", ms / n_iters);
    
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    printf("========================================\n");
    printf(" CUDA Kernel Fusion Benchmark\n");
    printf("========================================\n\n");
    
    int dim = 288, hidden_dim = 768;
    int n_iters = 10000;
    
    printf("Config: dim=%d hidden=%d iters=%d\n\n", dim, hidden_dim, n_iters);
    
    // Alocar buffers
    half *d_input, *d_residual, *d_gamma, *d_output;
    half *d_q, *d_k, *d_cos, *d_sin;
    half *d_gate, *d_up;
    
    CUDA_CHECK(cudaMalloc(&d_input, dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_residual, dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_gamma, dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_q, dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_k, dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_cos, dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_sin, dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_gate, hidden_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_up, hidden_dim * sizeof(half)));
    
    // Inicializar dados
    half* h_data = (half*)malloc(max(dim, hidden_dim) * sizeof(half));
    for (int i = 0; i < max(dim, hidden_dim); i++) 
        h_data[i] = __float2half((float)(rand() % 100) / 100.0f);
    
    CUDA_CHECK(cudaMemcpy(d_input, h_data, dim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_residual, h_data, dim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_data, dim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q, h_data, dim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_data, dim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cos, h_data, dim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sin, h_data, dim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gate, h_data, hidden_dim * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_up, h_data, hidden_dim * sizeof(half), cudaMemcpyHostToDevice));
    free(h_data);
    
    // Warmup
    fused_rmsnorm_add_kernel<<<1, 256, 256*sizeof(float)>>>(
        d_input, d_residual, d_gamma, d_output, dim, 1e-5f);
    cudaDeviceSynchronize();
    
    // Benchmark
    printf("Benchmarking...\n\n");
    
    // Nota: benchmark_unfused requer implementação completa dos kernels separados
    // Para demonstração, mostramos apenas o fused
    benchmark_fused(d_input, d_residual, d_gamma, d_output,
                    d_q, d_k, d_cos, d_sin, d_gate, d_up,
                    dim, hidden_dim, n_iters);
    
    printf("\n========================================\n");
    printf(" Kernel Fusion Benefits:\n");
    printf(" - Reduz kernel launch overhead\n");
    printf(" - Melhora memory locality\n");
    printf(" - Mantém dados em registers/shared mem\n");
    printf("========================================\n");
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_residual));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_cos));
    CUDA_CHECK(cudaFree(d_sin));
    CUDA_CHECK(cudaFree(d_gate));
    CUDA_CHECK(cudaFree(d_up));
    
    return 0;
}
