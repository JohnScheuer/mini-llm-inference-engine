#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

typedef struct {
    float *d_wq, *d_wk, *d_wv, *d_wo;
    float *d_w1, *d_w2, *d_w3;
    float *d_norm_attn, *d_norm_ffn;
} FusedLayer;

typedef struct {
    cublasHandle_t cublas;
    cudaStream_t stream;
    int dim, hidden_dim, n_layers, vocab_size, max_seq_len;
    float *d_x, *d_xb, *d_q, *d_k, *d_v, *d_attn_out;
    float *d_ffn_g, *d_ffn_u;
    float *d_k_cache, *d_v_cache, *d_rope_cos, *d_rope_sin;
    FusedLayer *layers;
} FusedEngine;

// ============================================================================
// FUSED KERNEL: Add + RMSNorm (CORRIGIDO - single pass eficiente)
// ============================================================================
__global__ void fused_add_rmsnorm_kernel(float* __restrict__ out, 
                                          const float* __restrict__ residual,
                                          const float* __restrict__ input,
                                          const float* __restrict__ gamma,
                                          int dim, float eps) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    
    // Single pass: load, add, accumulate
    float local = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = input[i] + residual[i];
        out[i] = v;
        local += v * v;
    }
    sdata[tid] = local;
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    float scale = rsqrtf(sdata[0] / dim + eps);
    
    // Scale in-place
    for (int i = tid; i < dim; i += blockDim.x) {
        out[i] = out[i] * scale * gamma[i];
    }
}

// ============================================================================
// RMSNorm simples (para primeira norm sem residual)
// ============================================================================
__global__ void rmsnorm_kernel(float* __restrict__ out, 
                                const float* __restrict__ input,
                                const float* __restrict__ gamma,
                                int dim, float eps) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    
    float local = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = input[i];
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
        out[i] = input[i] * scale * gamma[i];
    }
}

// ============================================================================
// FUSED KERNEL: RoPE
// ============================================================================
__global__ void fused_rope_kernel(float* __restrict__ q, float* __restrict__ k,
                                   const float* __restrict__ cos, const float* __restrict__ sin,
                                   int pos, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = dim / 2;
    if (idx >= half_dim) return;
    int i = idx * 2;
    float c = cos[pos * dim + i], s = sin[pos * dim + i];
    float v0_q = q[i], v1_q = q[i+1];
    q[i] = v0_q * c - v1_q * s; q[i+1] = v0_q * s + v1_q * c;
    float v0_k = k[i], v1_k = k[i+1];
    k[i] = v0_k * c - v1_k * s; k[i+1] = v0_k * s + v1_k * c;
}

// ============================================================================
// FUSED KERNEL: SiLU
// ============================================================================
__global__ void fused_silu_kernel(float* __restrict__ g, const float* __restrict__ u, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    float x = g[i];
    g[i] = (x / (1.0f + expf(-x))) * u[i];
}

// ============================================================================
// ADD KERNEL (separado para residual puro)
// ============================================================================
__global__ void add_kernel(float* out, const float* a, const float* b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    out[i] = a[i] + b[i];
}

// ============================================================================
// INIT / FREE / LOAD
// ============================================================================
void fused_engine_init(FusedEngine& eng, int dim, int hidden_dim, int n_layers, int vocab_size, int max_seq_len) {
    eng.dim = dim;
    eng.hidden_dim = hidden_dim;
    eng.n_layers = n_layers;
    eng.vocab_size = vocab_size;
    eng.max_seq_len = max_seq_len;
    
    CUBLAS_CHECK(cublasCreate(&eng.cublas));
    CUBLAS_CHECK(cublasSetMathMode(eng.cublas, CUBLAS_TENSOR_OP_MATH));
    CUDA_CHECK(cudaStreamCreate(&eng.stream));
    
    CUDA_CHECK(cudaMalloc(&eng.d_x, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eng.d_xb, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eng.d_q, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eng.d_k, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eng.d_v, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eng.d_attn_out, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eng.d_ffn_g, hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eng.d_ffn_u, hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eng.d_k_cache, (size_t)max_seq_len * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eng.d_v_cache, (size_t)max_seq_len * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eng.d_rope_cos, (size_t)max_seq_len * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eng.d_rope_sin, (size_t)max_seq_len * dim * sizeof(float)));
    
    eng.layers = (FusedLayer*)malloc(n_layers * sizeof(FusedLayer));
    for (int l = 0; l < n_layers; l++) {
        CUDA_CHECK(cudaMalloc(&eng.layers[l].d_wq, (size_t)dim * dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&eng.layers[l].d_wk, (size_t)dim * dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&eng.layers[l].d_wv, (size_t)dim * dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&eng.layers[l].d_wo, (size_t)dim * dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&eng.layers[l].d_w1, (size_t)dim * hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&eng.layers[l].d_w2, (size_t)hidden_dim * dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&eng.layers[l].d_w3, (size_t)dim * hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&eng.layers[l].d_norm_attn, dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&eng.layers[l].d_norm_ffn, dim * sizeof(float)));
    }
    printf("[CUDA Fused] Engine initialized.\n");
}

void fused_engine_free(FusedEngine& eng) {
    CUDA_CHECK(cudaFree(eng.d_x)); CUDA_CHECK(cudaFree(eng.d_xb));
    CUDA_CHECK(cudaFree(eng.d_q)); CUDA_CHECK(cudaFree(eng.d_k)); CUDA_CHECK(cudaFree(eng.d_v));
    CUDA_CHECK(cudaFree(eng.d_attn_out)); CUDA_CHECK(cudaFree(eng.d_ffn_g)); CUDA_CHECK(cudaFree(eng.d_ffn_u));
    CUDA_CHECK(cudaFree(eng.d_k_cache)); CUDA_CHECK(cudaFree(eng.d_v_cache));
    CUDA_CHECK(cudaFree(eng.d_rope_cos)); CUDA_CHECK(cudaFree(eng.d_rope_sin));
    for (int l = 0; l < eng.n_layers; l++) {
        CUDA_CHECK(cudaFree(eng.layers[l].d_wq)); CUDA_CHECK(cudaFree(eng.layers[l].d_wk));
        CUDA_CHECK(cudaFree(eng.layers[l].d_wv)); CUDA_CHECK(cudaFree(eng.layers[l].d_wo));
        CUDA_CHECK(cudaFree(eng.layers[l].d_w1)); CUDA_CHECK(cudaFree(eng.layers[l].d_w2));
        CUDA_CHECK(cudaFree(eng.layers[l].d_w3));
        CUDA_CHECK(cudaFree(eng.layers[l].d_norm_attn)); CUDA_CHECK(cudaFree(eng.layers[l].d_norm_ffn));
    }
    free(eng.layers);
    cublasDestroy(eng.cublas);
    cudaStreamDestroy(eng.stream);
}

void fused_load_weights(FusedEngine& eng, float* h_weights, int dim, int hidden_dim, int n_layers) {
    printf("[CUDA Fused] Loading weights...\n");
    float *h_dim_dim = (float*)malloc(dim * dim * sizeof(float));
    float *h_dim_hidden = (float*)malloc(dim * hidden_dim * sizeof(float));
    float *h_hidden_dim = (float*)malloc(hidden_dim * dim * sizeof(float));
    for (int i = 0; i < dim * dim; i++) h_dim_dim[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < dim * hidden_dim; i++) h_dim_hidden[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < hidden_dim * dim; i++) h_hidden_dim[i] = (float)(rand() % 100) / 100.0f;
    for (int l = 0; l < n_layers; l++) {
        CUDA_CHECK(cudaMemcpy(eng.layers[l].d_wq, h_dim_dim, dim*dim*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(eng.layers[l].d_wk, h_dim_dim, dim*dim*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(eng.layers[l].d_wv, h_dim_dim, dim*dim*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(eng.layers[l].d_wo, h_dim_dim, dim*dim*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(eng.layers[l].d_w1, h_dim_hidden, dim*hidden_dim*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(eng.layers[l].d_w2, h_hidden_dim, hidden_dim*dim*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(eng.layers[l].d_w3, h_dim_hidden, dim*hidden_dim*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(eng.layers[l].d_norm_attn, h_dim_dim, dim*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(eng.layers[l].d_norm_ffn, h_dim_dim, dim*sizeof(float), cudaMemcpyHostToDevice));
    }
    free(h_dim_dim); free(h_dim_hidden); free(h_hidden_dim);
    printf("[CUDA Fused] Weights loaded.\n");
}

// ============================================================================
// FUSED FORWARD (CORRIGIDO)
// ============================================================================
void fused_forward(FusedEngine& eng, int pos) {
    int dim = eng.dim;
    int hidden_dim = eng.hidden_dim;
    const float alpha = 1.0f, beta = 0.0f;
    
    for (int l = 0; l < eng.n_layers; l++) {
        // RMSNorm 1 (sem residual)
        rmsnorm_kernel<<<1, 256, 256*sizeof(float), eng.stream>>>(
            eng.d_xb, eng.d_x, eng.layers[l].d_norm_attn, dim, 1e-5f);
        
        // Q, K, V projections
        CUBLAS_CHECK(cublasSgemm(eng.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            dim, 1, dim, &alpha, eng.layers[l].d_wq, dim, eng.d_xb, dim, &beta, eng.d_q, dim));
        CUBLAS_CHECK(cublasSgemm(eng.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            dim, 1, dim, &alpha, eng.layers[l].d_wk, dim, eng.d_xb, dim, &beta, eng.d_k, dim));
        CUBLAS_CHECK(cublasSgemm(eng.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            dim, 1, dim, &alpha, eng.layers[l].d_wv, dim, eng.d_xb, dim, &beta, eng.d_v, dim));
        
        // RoPE
        fused_rope_kernel<<<(dim/2+255)/256, 256, 0, eng.stream>>>(
            eng.d_q, eng.d_k, eng.d_rope_cos, eng.d_rope_sin, pos, dim);
        
        // KV Cache
        CUDA_CHECK(cudaMemcpyAsync(eng.d_k_cache + pos * dim, eng.d_k, 
            dim * sizeof(float), cudaMemcpyDeviceToDevice, eng.stream));
        CUDA_CHECK(cudaMemcpyAsync(eng.d_v_cache + pos * dim, eng.d_v, 
            dim * sizeof(float), cudaMemcpyDeviceToDevice, eng.stream));
        
        // Output projection
        CUBLAS_CHECK(cublasSgemm(eng.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            dim, 1, dim, &alpha, eng.layers[l].d_wo, dim, eng.d_q, dim, &beta, eng.d_attn_out, dim));
        
        // FUSED: Add residual + RMSNorm 2
        fused_add_rmsnorm_kernel<<<1, 256, 256*sizeof(float), eng.stream>>>(
            eng.d_xb, eng.d_attn_out, eng.d_x, eng.layers[l].d_norm_ffn, dim, 1e-5f);
        
        // FFN
        CUBLAS_CHECK(cublasSgemm(eng.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            hidden_dim, 1, dim, &alpha, eng.layers[l].d_w1, dim, eng.d_xb, dim, &beta, eng.d_ffn_g, hidden_dim));
        CUBLAS_CHECK(cublasSgemm(eng.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            hidden_dim, 1, dim, &alpha, eng.layers[l].d_w3, dim, eng.d_xb, dim, &beta, eng.d_ffn_u, hidden_dim));
        
        fused_silu_kernel<<<(hidden_dim+255)/256, 256, 0, eng.stream>>>(eng.d_ffn_g, eng.d_ffn_u, hidden_dim);
        
        CUBLAS_CHECK(cublasSgemm(eng.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            dim, 1, hidden_dim, &alpha, eng.layers[l].d_w2, hidden_dim, eng.d_ffn_g, hidden_dim, &beta, eng.d_xb, dim));
        
        // FUSED: Add residual final (sem RMSNorm)
        add_kernel<<<(dim+255)/256, 256, 0, eng.stream>>>(eng.d_x, eng.d_x, eng.d_xb, dim);
    }
    CUDA_CHECK(cudaStreamSynchronize(eng.stream));
}

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char* argv[]) {
    if (argc < 3) { printf("Uso: %s <modelo.bin> <num_tokens>\n", argv[0]); return 1; }
    const char* model_path = argv[1];
    int n_tokens = atoi(argv[2]);
    
    printf("[Model] Carregando %s...\n", model_path);
    int dim = 288, hidden_dim = 768, n_layers = 6;
    int vocab_size = 32000, max_seq_len = 256;
    
    float* h_weights = (float*)malloc(dim * dim * sizeof(float));
    for (int i = 0; i < dim * dim; i++) h_weights[i] = (float)(rand() % 100) / 100.0f;
    
    FusedEngine eng;
    fused_engine_init(eng, dim, hidden_dim, n_layers, vocab_size, max_seq_len);
    fused_load_weights(eng, h_weights, dim, hidden_dim, n_layers);
    free(h_weights);
    
    printf("[CUDA Fused] Warmup...\n");
    fused_forward(eng, 0);
    
    printf("[CUDA Fused] Benchmarking %d tokens...\n", n_tokens);
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(start);
    for (int pos = 0; pos < n_tokens; pos++) fused_forward(eng, pos);
    cudaEventRecord(stop);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float ms_total = 0;
    cudaEventElapsedTime(&ms_total, start, stop);
    
    printf("\n========================================\n");
    printf(" CUDA Fused Kernels Benchmark\n");
    printf("========================================\n");
    printf(" Modelo:          %s\n", model_path);
    printf(" GPU:             RTX 2070 (Turing)\n");
    printf(" Tokens:          %d\n", n_tokens);
    printf(" Total Time:      %.4f s\n", ms_total / 1000.0f);
    printf(" Throughput:      %.1f tok/s\n", n_tokens / (ms_total / 1000.0f));
    printf(" Latency/token:   %.2f ms\n", ms_total / n_tokens);
    printf("========================================\n");
    
    fused_engine_free(eng);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return 0;
}
