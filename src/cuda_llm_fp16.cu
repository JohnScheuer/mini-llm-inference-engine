#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <iostream>
#include "json.hpp"
#include "model.h"
#include "tokenizer.h"

using namespace nvcuda;
using json = nlohmann::json;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(1); } } while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t s = call; \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS error: %d\n", s); exit(1); } } while(0)

// ============================================================
// Kernels Otimizados
// ============================================================

__global__ void init_rng_kernel(curandState* states, unsigned long seed) {
    curand_init(seed, 0, 0, &states[0]);
}

__global__ void rmsnorm_kernel(half* out, const half* x, const half* gamma, int dim, float eps) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x; float local = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) { float xi = __half2float(x[i]); local += xi * xi; }
    sdata[tid] = local; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) { if (tid < s) sdata[tid] += sdata[tid + s]; __syncthreads(); }
    float scale = rsqrtf(sdata[0] / dim + eps);
    for (int i = tid; i < dim; i += blockDim.x) out[i] = __float2half(__half2float(x[i]) * scale * __half2float(gamma[i]));
}

__global__ void residual_rmsnorm_fused_kernel(half* out, half* x, const half* residual, const half* gamma, int dim, float eps) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x; float local_sq_sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float val = __half2float(x[i]) + __half2float(residual[i]);
        x[i] = __float2half(val);
        local_sq_sum += val * val;
    }
    sdata[tid] = local_sq_sum; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) { if (tid < s) sdata[tid] += sdata[tid + s]; __syncthreads(); }
    float scale = rsqrtf(sdata[0] / dim + eps);
    for (int i = tid; i < dim; i += blockDim.x) out[i] = __float2half(__half2float(x[i]) * scale * __half2float(gamma[i]));
}

__global__ void silu_fused_kernel(half* w13_res, int hidden_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden_dim / 2) return;
    half2* g2 = reinterpret_cast<half2*>(w13_res);
    const half2* u2 = reinterpret_cast<const half2*>(w13_res + hidden_dim);
    float2 x = __half22float2(g2[i]); float2 y = __half22float2(u2[i]);
    x.x = (x.x / (1.0f + expf(-x.x))) * y.x; x.y = (x.y / (1.0f + expf(-x.y))) * y.y;
    g2[i] = __float22half2_rn(x);
}

__global__ void residual_kernel_v2(half* a, const half* b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size / 2) {
        half2* a2 = reinterpret_cast<half2*>(a);
        const half2* b2 = reinterpret_cast<const half2*>(b);
        a2[i] = __hadd2(a2[i], b2[i]);
    }
}

// CORREÇÃO: RoPE Split compatível diretamente com o formato HuggingFace das Safetensors
__global__ void rope_split_kernel_v2(half* q, half* k, const float* cos_ptr, const float* sin_ptr, int dim, int n_heads) {
    int h_dim = dim / n_heads;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim / 2) return;
    
    int head = idx / (h_dim / 2);
    int tid = idx % (h_dim / 2);
    
    int idx0 = head * h_dim + tid;
    int idx1 = head * h_dim + tid + h_dim / 2;
    
    float c = cos_ptr[idx0];
    float s = sin_ptr[idx0];
    
    auto rot = [&](half* vec) {
        float v0 = __half2float(vec[idx0]);
        float v1 = __half2float(vec[idx1]);
        vec[idx0] = __float2half(v0 * c - v1 * s);
        vec[idx1] = __float2half(v1 * c + v0 * s);
    };
    rot(q); rot(k);
}

__global__ void embedding_kernel_v2(half* out, const half* table, int token, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim / 2) {
        half2* out2 = reinterpret_cast<half2*>(out);
        const half2* table2 = reinterpret_cast<const half2*>(table);
        out2[i] = table2[(token * dim / 2) + i];
    }
}

__global__ void sample_kernel(int* next, const float* logits, curandState* rng, float temp, int vocab_size) {
    if (threadIdx.x == 0) {
        float mv = -1e30f; for(int i=0; i<vocab_size; i++) mv = fmaxf(mv, logits[i]);
        float se = 0.0f; for(int i=0; i<vocab_size; i++) se += expf((logits[i] - mv) / temp);
        float r = curand_uniform(rng) * se; float acc = 0.0f; int sel = vocab_size - 1;
        for(int i=0; i<vocab_size; i++) { acc += expf((logits[i] - mv) / temp); if(acc >= r) { sel = i; break; } }
        *next = sel;
    }
}

__global__ void flash_attention_2d_kernel(const half* q, const half* k_cache, const half* v_cache, half* out, float* p_o, float* p_m, float* p_l, int seq_len, int head_dim, int n_heads, float scale) {
    int head = blockIdx.x; int tile = blockIdx.y; int tid = threadIdx.x;
    int t_size = 16; int t_start = tile * t_size; int t_end = min(t_start + t_size, seq_len);
    if (t_start >= seq_len) return;
    const half* q_ptr = q + head * head_dim;
    float q_val = (tid < head_dim) ? __half2float(q_ptr[tid]) : 0.f;
    float m_i = -1e30f, l_i = 0.f, o_i = 0.f;
    for (int t = t_start; t < t_end; t++) {
        const half* k_h = k_cache + (t * n_heads * head_dim) + (head * head_dim);
        const half* v_h = v_cache + (t * n_heads * head_dim) + (head * head_dim);
        float prod = (tid < head_dim) ? q_val * __half2float(k_h[tid]) : 0.f;
        for (int s = 16; s > 0; s >>= 1) prod += __shfl_down_sync(0xffffffff, prod, s);
        float score = __shfl_sync(0xffffffff, prod, 0) * scale;
        float n_m = fmaxf(m_i, score); float c_o = expf(m_i - n_m); float c_n = expf(score - n_m);
        l_i = l_i * c_o + c_n; m_i = n_m;
        if (tid < head_dim) o_i = o_i * c_o + c_n * __half2float(v_h[tid]);
    }
    int idx = head * gridDim.y + tile;
    if (tid < head_dim) p_o[idx * head_dim + tid] = o_i;
    if (tid == 0) { p_m[idx] = m_i; p_l[idx] = l_i; }
}

__global__ void flash_reduce_kernel(const float* p_o, const float* p_m, const float* p_l, half* out, int head_dim, int num_tiles) {
    int head = blockIdx.x; int tid = threadIdx.x; if (tid >= head_dim) return;
    float g_m = -1e30f, g_l = 0.f, g_o = 0.f;
    for (int t = 0; t < num_tiles; t++) {
        int idx = head * num_tiles + t;
        float m = p_m[idx], l = p_l[idx], o = p_o[idx * head_dim + tid];
        float n_m = fmaxf(g_m, m); float e_o = expf(g_m - n_m), e_n = expf(m - n_m);
        g_l = g_l * e_o + l * e_n; g_o = g_o * e_o + o * e_n; g_m = n_m;
    }
    out[head * head_dim + tid] = __float2half(g_o / g_l);
}

// ============================================================
// Orquestração Host & Estruturas
// ============================================================

struct CudaLayer { half *d_wqkv, *d_wo, *d_w13, *d_w2, *d_norm_attn, *d_norm_ffn; };
struct CudaModel { half *d_token_embedding, *d_norm_final, *d_lm_head; CudaLayer* layers; int n_layers; };
struct CudaRunState { half *d_x, *d_xb, *d_attn_out, *d_ffn_out, *d_qkv_res, *d_w13_res, *d_k_cache, *d_v_cache, *d_zero_vec; float *d_logits, *d_rope_cos, *d_rope_sin; curandState *d_rng_states; cublasHandle_t cublas; float *p_o, *p_m, *p_l; };

void load_safetensors_final(const std::string& path, Model& m) {
    std::ifstream f(path, std::ios::binary); if (!f) exit(1);
    uint64_t hs; f.read((char*)&hs, 8); std::string hj(hs, ' '); f.read(&hj[0], hs);
    json meta = json::parse(hj); uint64_t off = 8 + hs;
    auto read_t = [&](std::string name) {
        if (!meta.contains(name)) return std::vector<float>();
        std::string dtype = meta[name]["dtype"];
        uint64_t s = meta[name]["data_offsets"][0], e = meta[name]["data_offsets"][1];
        f.seekg(off + s); std::vector<uint16_t> h((e-s)/2); f.read((char*)h.data(), e-s);
        std::vector<float> out(h.size());
        for(size_t i=0; i<h.size(); i++) {
            if (dtype == "BF16") { uint32_t fv = (uint32_t)h[i] << 16; std::memcpy(&out[i], &fv, 4); }
            else { 
                unsigned short val = h[i]; uint32_t sign = (val & 0x8000) << 16; uint32_t exp = (val & 0x7c00) >> 10; uint32_t mant = (val & 0x03ff) << 13;
                if (exp == 0x1f) exp = 0xff; else if (exp != 0) exp += 127 - 15;
                uint32_t fv = sign | (exp << 23) | mant; std::memcpy(&out[i], &fv, 4);
            }
        }
        return out;
    };
    m.token_embedding.data = read_t("model.embed_tokens.weight");
    int h_dim = m.dim / m.n_heads;
    for(int i=0; i<m.n_layers; i++) {
        std::string p = "model.layers." + std::to_string(i);
        m.layers[i].norm_attn.data = read_t(p + ".input_layernorm.weight");
        m.layers[i].norm_ffn.data = read_t(p + ".post_attention_layernorm.weight");
        m.layers[i].wq.data = read_t(p + ".self_attn.q_proj.weight");
        auto wk_raw = read_t(p + ".self_attn.k_proj.weight"); auto wv_raw = read_t(p + ".self_attn.v_proj.weight");
        m.layers[i].wk.data.resize(m.dim * m.dim); m.layers[i].wv.data.resize(m.dim * m.dim);
        for(int qh=0; qh<32; qh++) {
            std::memcpy(&m.layers[i].wk.data[qh*h_dim*m.dim], &wk_raw[(qh/8)*h_dim*m.dim], h_dim*m.dim*4);
            std::memcpy(&m.layers[i].wv.data[qh*h_dim*m.dim], &wv_raw[(qh/8)*h_dim*m.dim], h_dim*m.dim*4);
        }
        m.layers[i].wo.data = read_t(p + ".self_attn.o_proj.weight");
        m.layers[i].w1.data = read_t(p + ".mlp.gate_proj.weight");
        m.layers[i].w2.data = read_t(p + ".mlp.down_proj.weight");
        m.layers[i].w3.data = read_t(p + ".mlp.up_proj.weight");
    }
    m.norm_final.data = read_t("model.norm.weight"); m.lm_head.data = read_t("lm_head.weight");
}

void cuda_load_model(CudaModel& cm, const Model& m) {
    auto to_gpu = [&](const std::vector<float>& h, size_t n) {
        half* d; CUDA_CHECK(cudaMalloc(&d, n * 2));
        std::vector<half> tmp(n); for(size_t i=0; i<n; i++) tmp[i] = __float2half(h[i]);
        CUDA_CHECK(cudaMemcpy(d, tmp.data(), n * 2, cudaMemcpyHostToDevice)); return d;
    };
    cm.n_layers = m.n_layers; cm.d_token_embedding = to_gpu(m.token_embedding.data, m.vocab_size * m.dim);
    cm.d_norm_final = to_gpu(m.norm_final.data, m.dim); cm.d_lm_head = to_gpu(m.lm_head.data, m.vocab_size * m.dim);
    cm.layers = new CudaLayer[m.n_layers];
    for(int i=0; i<m.n_layers; i++) {
        auto& l = m.layers[i]; auto& cl = cm.layers[i];
        std::vector<float> h_qkv(3*m.dim*m.dim), h_w13(2*m.dim*m.hidden_dim);
        std::memcpy(h_qkv.data(), l.wq.data.data(), m.dim*m.dim*4); std::memcpy(h_qkv.data()+m.dim*m.dim, l.wk.data.data(), m.dim*m.dim*4); std::memcpy(h_qkv.data()+2*m.dim*m.dim, l.wv.data.data(), m.dim*m.dim*4);
        std::memcpy(h_w13.data(), l.w1.data.data(), m.dim*m.hidden_dim*4); std::memcpy(h_w13.data()+m.dim*m.hidden_dim, l.w3.data.data(), m.dim*m.hidden_dim*4);
        cl.d_wqkv = to_gpu(h_qkv, 3*m.dim*m.dim); cl.d_w13 = to_gpu(h_w13, 2*m.dim*m.hidden_dim);
        cl.d_wo = to_gpu(l.wo.data, m.dim*m.dim); cl.d_w2 = to_gpu(l.w2.data, m.hidden_dim*m.dim);
        cl.d_norm_attn = to_gpu(l.norm_attn.data, m.dim); cl.d_norm_ffn = to_gpu(l.norm_ffn.data, m.dim);
    }
}

void cuda_init_run_state(CudaRunState& cs, const Model& m) {
    auto alloc_h = [&](size_t n) { half* d; CUDA_CHECK(cudaMalloc(&d, n*2)); return d; };
    auto alloc_f = [&](size_t n) { float* d; CUDA_CHECK(cudaMalloc(&d, n*4)); return d; };
    cs.d_x = alloc_h(m.dim); cs.d_xb = alloc_h(m.dim); cs.d_attn_out = alloc_h(m.dim); cs.d_ffn_out = alloc_h(m.dim);
    cs.d_qkv_res = alloc_h(3*m.dim); cs.d_w13_res = alloc_h(2*m.hidden_dim); cs.d_zero_vec = alloc_h(m.dim);
    CUDA_CHECK(cudaMemset(cs.d_zero_vec, 0, m.dim*2));
    cs.d_k_cache = alloc_h(m.max_seq_len * m.dim); cs.d_v_cache = alloc_h(m.max_seq_len * m.dim);
    cs.d_logits = alloc_f(m.vocab_size); cs.d_rope_cos = alloc_f(m.max_seq_len * m.dim); cs.d_rope_sin = alloc_f(m.max_seq_len * m.dim);
    
    // CORREÇÃO: Tabela de Cosseno e Seno com repetição exata e wrap-around por cabeça
    std::vector<float> cb(m.max_seq_len*m.dim), sb(m.max_seq_len*m.dim);
    int h_dim = m.dim / m.n_heads;
    for(int p=0; p<m.max_seq_len; p++) {
        for(int h=0; h<m.n_heads; h++) {
            for(int i=0; i<h_dim/2; i++) {
                float f = 1.0f / powf(10000.f, (float)(i*2)/h_dim);
                float v = (float)p * f;
                cb[p*m.dim + h*h_dim + i] = cosf(v);
                cb[p*m.dim + h*h_dim + i + h_dim/2] = cosf(v);
                sb[p*m.dim + h*h_dim + i] = sinf(v);
                sb[p*m.dim + h*h_dim + i + h_dim/2] = sinf(v);
            }
        }
    }
    CUDA_CHECK(cudaMemcpy(cs.d_rope_cos, cb.data(), m.max_seq_len*m.dim*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cs.d_rope_sin, sb.data(), m.max_seq_len*m.dim*4, cudaMemcpyHostToDevice));
    
    int max_tiles = (m.max_seq_len + 15) / 16;
    cs.p_o = alloc_f(m.n_heads * max_tiles * (m.dim/m.n_heads)); cs.p_m = alloc_f(m.n_heads * max_tiles); cs.p_l = alloc_f(m.n_heads * max_tiles);
    
    CUBLAS_CHECK(cublasCreate(&cs.cublas));
    CUBLAS_CHECK(cublasSetMathMode(cs.cublas, CUBLAS_TENSOR_OP_MATH));
}

void cuda_forward(CudaRunState& cs, const CudaModel& cm, const Model& m, int token, int pos) {
    int dim = m.dim, h_dim = dim/m.n_heads, hidden_dim = m.hidden_dim, seq_len = pos+1; float alpha = 1.0f, beta = 0.0f;
    embedding_kernel_v2<<<(dim/2+255)/256, 256>>>(cs.d_x, cm.d_token_embedding, token, dim);
    for (int l = 0; l < m.n_layers; l++) {
        auto& cl = cm.layers[l];
        residual_rmsnorm_fused_kernel<<<1, 256, 1024>>>(cs.d_xb, cs.d_x, (l==0 ? cs.d_zero_vec : cs.d_attn_out), cl.d_norm_attn, dim, 1e-5f);
        CUBLAS_CHECK(cublasGemmEx(cs.cublas, CUBLAS_OP_T, CUBLAS_OP_N, 3*dim, 1, dim, &alpha, cl.d_wqkv, CUDA_R_16F, dim, cs.d_xb, CUDA_R_16F, dim, &beta, cs.d_qkv_res, CUDA_R_16F, 3*dim, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        
        // CORREÇÃO: rope_split_kernel_v2 substituindo o intercalado antigo
        rope_split_kernel_v2<<<(dim/2+255)/256, 256>>>(cs.d_qkv_res, cs.d_qkv_res+dim, cs.d_rope_cos+pos*dim, cs.d_rope_sin+pos*dim, dim, m.n_heads);
        
        CUDA_CHECK(cudaMemcpy(cs.d_k_cache+pos*dim, cs.d_qkv_res+dim, dim*2, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(cs.d_v_cache+pos*dim, cs.d_qkv_res+2*dim, dim*2, cudaMemcpyDeviceToDevice));
        int tiles = (seq_len + 15) / 16;
        flash_attention_2d_kernel<<<dim3(m.n_heads, tiles), 64>>>(cs.d_qkv_res, cs.d_k_cache, cs.d_v_cache, cs.d_attn_out, cs.p_o, cs.p_m, cs.p_l, seq_len, h_dim, m.n_heads, 1.0f/sqrtf(h_dim));
        flash_reduce_kernel<<<m.n_heads, 64>>>(cs.p_o, cs.p_m, cs.p_l, cs.d_attn_out, h_dim, tiles);
        CUBLAS_CHECK(cublasGemmEx(cs.cublas, CUBLAS_OP_T, CUBLAS_OP_N, dim, 1, dim, &alpha, cl.d_wo, CUDA_R_16F, dim, cs.d_attn_out, CUDA_R_16F, dim, &beta, cs.d_xb, CUDA_R_16F, dim, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        residual_rmsnorm_fused_kernel<<<1, 256, 1024>>>(cs.d_xb, cs.d_x, cs.d_xb, cl.d_norm_ffn, dim, 1e-5f);
        CUBLAS_CHECK(cublasGemmEx(cs.cublas, CUBLAS_OP_T, CUBLAS_OP_N, 2*hidden_dim, 1, dim, &alpha, cl.d_w13, CUDA_R_16F, dim, cs.d_xb, CUDA_R_16F, dim, &beta, cs.d_w13_res, CUDA_R_16F, 2*hidden_dim, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        silu_fused_kernel<<<(hidden_dim/2+255)/256, 256>>>(cs.d_w13_res, hidden_dim);
        CUBLAS_CHECK(cublasGemmEx(cs.cublas, CUBLAS_OP_T, CUBLAS_OP_N, dim, 1, hidden_dim, &alpha, cl.d_w2, CUDA_R_16F, hidden_dim, cs.d_w13_res, CUDA_R_16F, hidden_dim, &beta, cs.d_attn_out, CUDA_R_16F, dim, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    residual_kernel_v2<<<(dim/2 + 255)/256, 256>>>(cs.d_x, cs.d_attn_out, dim);
    rmsnorm_kernel<<<1, 256, 1024>>>(cs.d_xb, cs.d_x, cm.d_norm_final, dim, 1e-5f);
    CUBLAS_CHECK(cublasGemmEx(cs.cublas, CUBLAS_OP_T, CUBLAS_OP_N, m.vocab_size, 1, dim, &alpha, cm.d_lm_head, CUDA_R_16F, dim, cs.d_xb, CUDA_R_16F, dim, &beta, cs.d_logits, CUDA_R_32F, m.vocab_size, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// ============================================================
// MAIN LOOP
// ============================================================

int main(int argc, char* argv[]) {
    if (argc < 3) return 1;
    Tokenizer tokenizer; if (!tokenizer.load(argv[2])) return 1;
    Model m; m.dim = 2048; m.hidden_dim = 5632; m.n_layers = 22; m.n_heads = 32; m.vocab_size = 32000; m.max_seq_len = 2048; m.layers.resize(m.n_layers);
    printf("[Loader] Carregando pesos...\n"); load_safetensors_final(argv[1], m);
    
    CudaModel cm; cuda_load_model(cm, m); 
    CudaRunState cs; cuda_init_run_state(cs, m);
    
    int* d_next;
    CUDA_CHECK(cudaMalloc(&d_next, 4));
    CUDA_CHECK(cudaMalloc(&cs.d_rng_states, sizeof(curandState))); 
    init_rng_kernel<<<1, 1>>>(cs.d_rng_states, 42);

    std::string p_str = (argc > 3) ? argv[3] : "Once upon a time";
    std::vector<int> tokens = tokenizer.encode(p_str); tokens.insert(tokens.begin(), 1);
    printf("\n[Prompt] %s\n[AI Response] ", p_str.c_str()); fflush(stdout);

    int curr = 0;
    int prev_token = 1; 

    auto start = std::chrono::high_resolution_clock::now();

    for(int pos=0; pos < m.max_seq_len; pos++) {
        if(pos < (int)tokens.size()) {
            curr = tokens[pos];
        }
        
        cuda_forward(cs, cm, m, curr, pos);
        
        if(pos >= (int)tokens.size() - 1) {
            sample_kernel<<<1, 256>>>(d_next, cs.d_logits, cs.d_rng_states, 0.7f, m.vocab_size);
            int next; 
            CUDA_CHECK(cudaMemcpy(&next, d_next, 4, cudaMemcpyDeviceToHost)); 
            if(next == 2) break; // EOS

            printf("%s", tokenizer.decode(prev_token, next).c_str()); fflush(stdout); 
            prev_token = next;
            curr = next; 
        }
    }
    
    double elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
    printf("\n========================================\n Mini-LLM Native FP16 GPU Backend\n========================================\n");
    printf(" Desempenho Medido: %.2f tok/s\n", (m.max_seq_len) / elapsed);
    
    cudaFree(d_next);
    cudaFree(cs.d_rng_states);
    cublasDestroy(cs.cublas);
    return 0;
}