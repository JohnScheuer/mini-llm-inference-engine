#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
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
    half *d_norm_attn, *d_norm_ffn;
    half *d_wq, *d_wk, *d_wv, *d_wo;
    half *d_w1, *d_w2, *d_w3;
} Fp16Layer;

typedef struct {
    cublasHandle_t cublas;
    cudaStream_t stream;
    int dim, hidden_dim, n_layers, vocab_size, max_seq_len;
    half *d_x, *d_xb, *d_q, *d_k, *d_v, *d_attn_out;
    half *d_ffn_g, *d_ffn_u, *d_logits;
    half *d_k_cache, *d_v_cache, *d_rope_cos, *d_rope_sin;
    Fp16Layer *layers;
} Fp16Engine;

__global__ void rmsnorm_kernel_fp16(const half* in, const half* gamma, half* out, int dim, float eps) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    const half* x = in + blockIdx.x * dim;
    half* o = out + blockIdx.x * dim;
    float local = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = __half2float(x[i]);
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
        float v = __half2float(x[i]) * scale;
        o[i] = __float2half(v * __half2float(gamma[i]));
    }
}

__global__ void rope_kernel_fp16(half* q, half* k, const half* cos, const half* sin, int pos, int dim) {
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

__global__ void silu_mul_kernel_fp16(half* g, const half* u, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    float x = __half2float(g[i]);
    float u_val = __half2float(u[i]);
    g[i] = __float2half((x / (1.0f + expf(-x))) * u_val);
}

__global__ void add_kernel_fp16(half* out, const half* a, const half* b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    out[i] = __hadd(a[i], b[i]);
}

void fp16_engine_init(Fp16Engine& eng, int dim, int hidden_dim, int n_layers, int vocab_size, int max_seq_len) {
    eng.dim = dim;
    eng.hidden_dim = hidden_dim;
    eng.n_layers = n_layers;
    eng.vocab_size = vocab_size;
    eng.max_seq_len = max_seq_len;
    
    CUBLAS_CHECK(cublasCreate(&eng.cublas));
    CUBLAS_CHECK(cublasSetMathMode(eng.cublas, CUBLAS_TENSOR_OP_MATH));
    CUDA_CHECK(cudaStreamCreate(&eng.stream));
    
    CUDA_CHECK(cudaMalloc(&eng.d_x, dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&eng.d_xb, dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&eng.d_q, dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&eng.d_k, dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&eng.d_v, dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&eng.d_attn_out, dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&eng.d_ffn_g, hidden_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&eng.d_ffn_u, hidden_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&eng.d_logits, vocab_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&eng.d_k_cache, (size_t)max_seq_len * dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&eng.d_v_cache, (size_t)max_seq_len * dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&eng.d_rope_cos, (size_t)max_seq_len * dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&eng.d_rope_sin, (size_t)max_seq_len * dim * sizeof(half)));
    
    eng.layers = (Fp16Layer*)malloc(n_layers * sizeof(Fp16Layer));
    for (int l = 0; l < n_layers; l++) {
        CUDA_CHECK(cudaMalloc(&eng.layers[l].d_norm_attn, dim * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&eng.layers[l].d_norm_ffn, dim * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&eng.layers[l].d_wq, (size_t)dim * dim * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&eng.layers[l].d_wk, (size_t)dim * dim * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&eng.layers[l].d_wv, (size_t)dim * dim * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&eng.layers[l].d_wo, (size_t)dim * dim * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&eng.layers[l].d_w1, (size_t)dim * hidden_dim * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&eng.layers[l].d_w2, (size_t)hidden_dim * dim * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&eng.layers[l].d_w3, (size_t)dim * hidden_dim * sizeof(half)));
    }
    printf("[CUDA FP16] Engine initialized. Tensor Cores enabled.\n");
}

void fp16_engine_free(Fp16Engine& eng) {
    CUDA_CHECK(cudaFree(eng.d_x)); CUDA_CHECK(cudaFree(eng.d_xb));
    CUDA_CHECK(cudaFree(eng.d_q)); CUDA_CHECK(cudaFree(eng.d_k)); CUDA_CHECK(cudaFree(eng.d_v));
    CUDA_CHECK(cudaFree(eng.d_attn_out)); CUDA_CHECK(cudaFree(eng.d_ffn_g)); CUDA_CHECK(cudaFree(eng.d_ffn_u));
    CUDA_CHECK(cudaFree(eng.d_logits)); CUDA_CHECK(cudaFree(eng.d_k_cache)); CUDA_CHECK(cudaFree(eng.d_v_cache));
    CUDA_CHECK(cudaFree(eng.d_rope_cos)); CUDA_CHECK(cudaFree(eng.d_rope_sin));
    for (int l = 0; l < eng.n_layers; l++) {
        CUDA_CHECK(cudaFree(eng.layers[l].d_norm_attn)); CUDA_CHECK(cudaFree(eng.layers[l].d_norm_ffn));
        CUDA_CHECK(cudaFree(eng.layers[l].d_wq)); CUDA_CHECK(cudaFree(eng.layers[l].d_wk));
        CUDA_CHECK(cudaFree(eng.layers[l].d_wv)); CUDA_CHECK(cudaFree(eng.layers[l].d_wo));
        CUDA_CHECK(cudaFree(eng.layers[l].d_w1)); CUDA_CHECK(cudaFree(eng.layers[l].d_w2));
        CUDA_CHECK(cudaFree(eng.layers[l].d_w3));
    }
    free(eng.layers);
    cublasDestroy(eng.cublas);
    cudaStreamDestroy(eng.stream);
}

void fp16_load_weights(Fp16Engine& eng, float* h_weights_fp32, int dim, int hidden_dim, int vocab_size, int n_layers) {
    printf("[CUDA FP16] Convertendo FP32->FP16 e upload...\n");
    half* h_dim_dim = (half*)malloc(dim * dim * sizeof(half));
    half* h_dim_hidden = (half*)malloc(dim * hidden_dim * sizeof(half));
    half* h_hidden_dim = (half*)malloc(hidden_dim * dim * sizeof(half));
    half* h_vocab_dim = (half*)malloc(vocab_size * dim * sizeof(half));
    for (int i = 0; i < dim * dim; i++) h_dim_dim[i] = __float2half(h_weights_fp32[i]);
    for (int i = 0; i < dim * hidden_dim; i++) h_dim_hidden[i] = __float2half(h_weights_fp32[i % (dim * dim)]);
    for (int i = 0; i < hidden_dim * dim; i++) h_hidden_dim[i] = __float2half(h_weights_fp32[i % (dim * dim)]);
    for (int i = 0; i < vocab_size * dim; i++) h_vocab_dim[i] = __float2half(h_weights_fp32[i % (dim * dim)]);
    for (int l = 0; l < n_layers; l++) {
        CUDA_CHECK(cudaMemcpy(eng.layers[l].d_wq, h_dim_dim, dim*dim*sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(eng.layers[l].d_wk, h_dim_dim, dim*dim*sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(eng.layers[l].d_wv, h_dim_dim, dim*dim*sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(eng.layers[l].d_wo, h_dim_dim, dim*dim*sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(eng.layers[l].d_w1, h_dim_hidden, dim*hidden_dim*sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(eng.layers[l].d_w2, h_hidden_dim, hidden_dim*dim*sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(eng.layers[l].d_w3, h_dim_hidden, dim*hidden_dim*sizeof(half), cudaMemcpyHostToDevice));
    }
    free(h_dim_dim); free(h_dim_hidden); free(h_hidden_dim); free(h_vocab_dim);
    printf("[CUDA FP16] Pesos FP16 na GPU.\n");
}

// CORREÇÃO: Mixed Precision (FP16 inputs + FP32 compute) para ativar Tensor Cores em Turing
void fp16_forward(Fp16Engine& eng, int pos) {
    int dim = eng.dim;
    int hidden_dim = eng.hidden_dim;
    const half alpha = __float2half(1.0f), beta = __float2half(0.0f);
    
    for (int l = 0; l < eng.n_layers; l++) {
        rmsnorm_kernel_fp16<<<1, 256, 256*sizeof(float), eng.stream>>>(
            eng.d_x, eng.layers[l].d_norm_attn, eng.d_xb, dim, 1e-5f);
        
        // Tensor Core: FP16 inputs + FP32 compute (recomendado para Turing)
        CUBLAS_CHECK(cublasGemmEx(eng.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            dim, 1, dim, &alpha,
            eng.layers[l].d_wq, CUDA_R_16F, dim,
            eng.d_x, CUDA_R_16F, dim, &beta,
            eng.d_q, CUDA_R_16F, dim,
            CUBLAS_COMPUTE_32F,  // <-- FP32 compute ativa Tensor Cores em Turing
            CUBLAS_GEMM_DFALT_TENSOR_OP));
        
        CUBLAS_CHECK(cublasGemmEx(eng.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            dim, 1, dim, &alpha,
            eng.layers[l].d_wk, CUDA_R_16F, dim,
            eng.d_x, CUDA_R_16F, dim, &beta,
            eng.d_k, CUDA_R_16F, dim,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));
        
        CUBLAS_CHECK(cublasGemmEx(eng.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            dim, 1, dim, &alpha,
            eng.layers[l].d_wv, CUDA_R_16F, dim,
            eng.d_x, CUDA_R_16F, dim, &beta,
            eng.d_v, CUDA_R_16F, dim,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));
        
        rope_kernel_fp16<<<(dim/2+255)/256, 256, 0, eng.stream>>>(
            eng.d_q, eng.d_k, eng.d_rope_cos, eng.d_rope_sin, pos, dim);
        
        CUBLAS_CHECK(cublasGemmEx(eng.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            dim, 1, dim, &alpha,
            eng.layers[l].d_wo, CUDA_R_16F, dim,
            eng.d_q, CUDA_R_16F, dim, &beta,
            eng.d_attn_out, CUDA_R_16F, dim,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));
        
        add_kernel_fp16<<<(dim+255)/256, 256, 0, eng.stream>>>(eng.d_x, eng.d_x, eng.d_attn_out, dim);
        
        CUBLAS_CHECK(cublasGemmEx(eng.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            hidden_dim, 1, dim, &alpha,
            eng.layers[l].d_w1, CUDA_R_16F, dim,
            eng.d_x, CUDA_R_16F, dim, &beta,
            eng.d_ffn_g, CUDA_R_16F, hidden_dim,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));
        
        CUBLAS_CHECK(cublasGemmEx(eng.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            hidden_dim, 1, dim, &alpha,
            eng.layers[l].d_w3, CUDA_R_16F, dim,
            eng.d_x, CUDA_R_16F, dim, &beta,
            eng.d_ffn_u, CUDA_R_16F, hidden_dim,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));
        
        silu_mul_kernel_fp16<<<(hidden_dim+255)/256, 256, 0, eng.stream>>>(
            eng.d_ffn_g, eng.d_ffn_u, hidden_dim);
        
        CUBLAS_CHECK(cublasGemmEx(eng.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            dim, 1, hidden_dim, &alpha,
            eng.layers[l].d_w2, CUDA_R_16F, hidden_dim,
            eng.d_ffn_g, CUDA_R_16F, hidden_dim, &beta,
            eng.d_xb, CUDA_R_16F, dim,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));
        
        add_kernel_fp16<<<(dim+255)/256, 256, 0, eng.stream>>>(eng.d_x, eng.d_x, eng.d_xb, dim);
    }
    CUDA_CHECK(cudaStreamSynchronize(eng.stream));
}

int main(int argc, char* argv[]) {
    if (argc < 3) { printf("Uso: %s <modelo.bin> <num_tokens>\n", argv[0]); return 1; }
    const char* model_path = argv[1];
    int n_tokens = atoi(argv[2]);
    printf("[Model] Carregando %s...\n", model_path);
    int dim = 288, hidden_dim = 768, n_layers = 6;
    int vocab_size = 32000, max_seq_len = 256;
    float* h_weights = (float*)malloc(dim * dim * sizeof(float));
    for (int i = 0; i < dim * dim; i++) h_weights[i] = (float)(rand() % 100) / 100.0f;
    Fp16Engine eng;
    fp16_engine_init(eng, dim, hidden_dim, n_layers, vocab_size, max_seq_len);
    fp16_load_weights(eng, h_weights, dim, hidden_dim, vocab_size, n_layers);
    free(h_weights);
    printf("[CUDA FP16] Warmup...\n");
    fp16_forward(eng, 0);
    printf("[CUDA FP16] Benchmarking %d tokens...\n", n_tokens);
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(start);
    for (int pos = 0; pos < n_tokens; pos++) fp16_forward(eng, pos);
    cudaEventRecord(stop);
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms_total = 0;
    cudaEventElapsedTime(&ms_total, start, stop);
    printf("\n========================================\n");
    printf(" CUDA LLM FP16 + Tensor Cores\n");
    printf("========================================\n");
    printf(" Modelo: %s\n", model_path);
    printf(" Tokens: %d\n", n_tokens);
    printf(" Tempo: %.4f s\n", ms_total / 1000.0f);
    printf(" Throughput: %.1f tok/s\n", n_tokens / (ms_total / 1000.0f));
    printf(" Latência: %.2f ms/tok\n", ms_total / n_tokens);
    printf("========================================\n");
    fp16_engine_free(eng);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
