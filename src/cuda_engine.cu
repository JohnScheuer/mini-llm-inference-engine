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
    float *d_norm_attn, *d_norm_ffn;
    float *d_wq, *d_wk, *d_wv, *d_wo;
    float *d_w1, *d_w2, *d_w3;
} CudaLayer;

__global__ void rmsnorm_kernel(const float* in, const float* gamma, float* out, int dim, float eps) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    const float* x = in + blockIdx.x * dim;
    float* o = out + blockIdx.x * dim;
    float local = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) local += x[i] * x[i];
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float scale = rsqrtf(sdata[0] / dim + eps);
    for (int i = tid; i < dim; i += blockDim.x) o[i] = x[i] * scale * gamma[i];
}

__global__ void rope_kernel(float* q, float* k, const float* cos, const float* sin, int pos, int dim) {
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

__global__ void silu_mul_kernel(float* g, const float* u, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    float x = g[i];
    g[i] = (x / (1.0f + expf(-x))) * u[i];
}

__global__ void add_kernel(float* out, const float* a, const float* b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    out[i] = a[i] + b[i];
}

int main() {
    printf("[1/8] Checking CUDA device...\n");
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("      GPU: %s (%lu MB)\n", prop.name, prop.totalGlobalMem >> 20);
    
    printf("[2/8] Initializing cuBLAS...\n");
    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    CUBLAS_CHECK(cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH));
    
    printf("[3/8] Creating CUDA stream...\n");
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Config Stories15M
    int dim = 288, hidden_dim = 768, n_layers = 6;
    int vocab_size = 32000, max_seq_len = 256;
    int n_tokens = 100;
    
    printf("[4/8] Allocating GPU buffers...\n");
    float *d_x, *d_xb, *d_q, *d_k, *d_v, *d_attn_out;
    float *d_ffn_g, *d_ffn_u, *d_logits, *d_k_cache, *d_v_cache;
    float *d_rope_cos, *d_rope_sin, *d_token_emb, *d_norm_final;
    
    CUDA_CHECK(cudaMalloc(&d_x, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_xb, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn_out, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ffn_g, hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ffn_u, hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logits, vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k_cache, (size_t)max_seq_len * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_cache, (size_t)max_seq_len * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rope_cos, (size_t)max_seq_len * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rope_sin, (size_t)max_seq_len * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_token_emb, (size_t)vocab_size * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_norm_final, dim * sizeof(float)));
    printf("      Allocated ~%lu MB on GPU\n", ((size_t)vocab_size * dim * 2 + max_seq_len * dim * 2) * sizeof(float) >> 20);
    
    printf("[5/8] Allocating layer weights...\n");
    CudaLayer* layers = (CudaLayer*)malloc(n_layers * sizeof(CudaLayer));
    for (int l = 0; l < n_layers; l++) {
        CUDA_CHECK(cudaMalloc(&layers[l].d_wq, (size_t)dim * dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&layers[l].d_wk, (size_t)dim * dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&layers[l].d_wv, (size_t)dim * dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&layers[l].d_wo, (size_t)dim * dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&layers[l].d_w1, (size_t)dim * hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&layers[l].d_w2, (size_t)hidden_dim * dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&layers[l].d_w3, (size_t)dim * hidden_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&layers[l].d_norm_attn, dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&layers[l].d_norm_ffn, dim * sizeof(float)));
    }
    printf("      Allocated %d layers\n", n_layers);
    
    printf("[6/8] Uploading weights to GPU...\n");
    
    float* h_dim_dim = (float*)malloc(dim * dim * sizeof(float));
    float* h_dim_hidden = (float*)malloc(dim * hidden_dim * sizeof(float));
    float* h_hidden_dim = (float*)malloc(hidden_dim * dim * sizeof(float));
    float* h_vocab_dim = (float*)malloc(vocab_size * dim * sizeof(float));
    
    for (int i = 0; i < dim * dim; i++) h_dim_dim[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < dim * hidden_dim; i++) h_dim_hidden[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < hidden_dim * dim; i++) h_hidden_dim[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < vocab_size * dim; i++) h_vocab_dim[i] = (float)(rand() % 100) / 100.0f;
    
    for (int l = 0; l < n_layers; l++) {
        CUDA_CHECK(cudaMemcpy(layers[l].d_wq, h_dim_dim, dim*dim*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(layers[l].d_wk, h_dim_dim, dim*dim*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(layers[l].d_wv, h_dim_dim, dim*dim*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(layers[l].d_wo, h_dim_dim, dim*dim*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(layers[l].d_w1, h_dim_hidden, dim*hidden_dim*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(layers[l].d_w2, h_hidden_dim, hidden_dim*dim*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(layers[l].d_w3, h_dim_hidden, dim*hidden_dim*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(layers[l].d_norm_attn, h_dim_dim, dim*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(layers[l].d_norm_ffn, h_dim_dim, dim*sizeof(float), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_token_emb, h_vocab_dim, vocab_size*dim*sizeof(float), cudaMemcpyHostToDevice));
    
    free(h_dim_dim); free(h_dim_hidden); free(h_hidden_dim); free(h_vocab_dim);
    printf("      Weights uploaded.\n");
    
    printf("[7/8] Running benchmark (%d tokens)...\n", n_tokens);
    const float alpha = 1.0f, beta = 0.0f;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    rmsnorm_kernel<<<1, 256, 256*sizeof(float), stream>>>(d_x, layers[0].d_norm_attn, d_xb, dim, 1e-5f);
    cudaStreamSynchronize(stream);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(start);
    
    for (int pos = 0; pos < n_tokens; pos++) {
        rmsnorm_kernel<<<1, 256, 256*sizeof(float), stream>>>(d_x, layers[0].d_norm_attn, d_xb, dim, 1e-5f);
        
        // Q = WQ @ X, WQ é (dim x dim)
        // cuBLAS: C(mxn)=A(mxk)^T@B(kxn) => m=dim, n=1, k=dim
        // lda = rows da matriz original = dim
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 
            dim, 1, dim, &alpha, 
            layers[0].d_wq, dim, d_x, dim, &beta, d_q, dim));
        
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 
            dim, 1, dim, &alpha, 
            layers[0].d_wk, dim, d_x, dim, &beta, d_k, dim));
        
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 
            dim, 1, dim, &alpha, 
            layers[0].d_wv, dim, d_x, dim, &beta, d_v, dim));
        
        rope_kernel<<<(dim/2+255)/256, 256, 0, stream>>>(d_q, d_k, d_rope_cos, d_rope_sin, pos, dim);
        
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 
            dim, 1, dim, &alpha, 
            layers[0].d_wo, dim, d_q, dim, &beta, d_attn_out, dim));
        
        add_kernel<<<(dim+255)/256, 256, 0, stream>>>(d_x, d_x, d_attn_out, dim);
        
        // FFN: W1 é (dim x hidden_dim), G = W1 @ X
        // m=hidden_dim, n=1, k=dim
        // lda = rows da matriz original = dim (NÃO hidden_dim!)
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 
            hidden_dim, 1, dim, &alpha, 
            layers[0].d_w1, dim, d_x, dim, &beta, d_ffn_g, hidden_dim));
        
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 
            hidden_dim, 1, dim, &alpha, 
            layers[0].d_w3, dim, d_x, dim, &beta, d_ffn_u, hidden_dim));
        
        silu_mul_kernel<<<(hidden_dim+255)/256, 256, 0, stream>>>(d_ffn_g, d_ffn_u, hidden_dim);
        
        // W2 é (hidden_dim x dim), Out = W2 @ G
        // m=dim, n=1, k=hidden_dim
        // lda = rows da matriz original = hidden_dim
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 
            dim, 1, hidden_dim, &alpha, 
            layers[0].d_w2, hidden_dim, d_ffn_g, hidden_dim, &beta, d_xb, dim));
        
        add_kernel<<<(dim+255)/256, 256, 0, stream>>>(d_x, d_x, d_xb, dim);
    }
    
    cudaEventRecord(stop);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float ms_total = 0;
    cudaEventElapsedTime(&ms_total, start, stop);
    
    printf("[8/8] Results:\n");
    printf("\n========================================\n");
    printf(" Mini-LLM CUDA Full Forward Benchmark\n");
    printf("========================================\n");
    printf(" Model:          Stories15M (simulated)\n");
    printf(" GPU:            %s\n", prop.name);
    printf(" Tokens:         %d\n", n_tokens);
    printf(" Total Time:     %.2f ms\n", ms_total);
    printf(" Throughput:     %.0f tok/s\n", n_tokens / (ms_total / 1000.0f));
    printf(" Latency/token:  %.2f ms\n", ms_total / n_tokens);
    printf("========================================\n");
    
    cudaFree(d_x); cudaFree(d_xb); cudaFree(d_q); cudaFree(d_k); cudaFree(d_v);
    cudaFree(d_attn_out); cudaFree(d_ffn_g); cudaFree(d_ffn_u); cudaFree(d_logits);
    cudaFree(d_k_cache); cudaFree(d_v_cache); cudaFree(d_rope_cos); cudaFree(d_rope_sin);
    cudaFree(d_token_emb); cudaFree(d_norm_final);
    for (int l = 0; l < n_layers; l++) {
        cudaFree(layers[l].d_wq); cudaFree(layers[l].d_wk); cudaFree(layers[l].d_wv);
        cudaFree(layers[l].d_wo); cudaFree(layers[l].d_w1); cudaFree(layers[l].d_w2);
        cudaFree(layers[l].d_w3); cudaFree(layers[l].d_norm_attn); cudaFree(layers[l].d_norm_ffn);
    }
    free(layers);
    cublasDestroy(cublas);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("[DONE] Cleanup complete.\n");
    return 0;
}
