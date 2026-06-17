#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>

#include "model.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t stat = call; \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, stat); \
        exit(1); \
    } \
} while(0)

// ============================================================
// Kernels (agora todos operam em float, exceto RoPE que lê half)
// ============================================================
__global__ void rmsnorm_kernel(float* out, const float* x, const half* gamma, int dim, float eps) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float local = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) local += x[i] * x[i];
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) { if (tid < s) sdata[tid] += sdata[tid + s]; __syncthreads(); }
    float scale = rsqrtf(sdata[0] / dim + eps);
    for (int i = tid; i < dim; i += blockDim.x) out[i] = x[i] * scale * __half2float(gamma[i]);
}

__global__ void rope_kernel(float* q, float* k, const float* cos, const float* sin, int pos, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim/2) return;
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

__global__ void residual_kernel(float* a, const float* b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) a[i] += b[i];
}

__global__ void embedding_kernel(float* out, const half* table, int token, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) out[i] = __half2float(table[token * dim + i]);
}

__global__ void attn_scores_kernel(float* scores, const float* q, const float* k_cache, int head_idx, int head_dim, int dim, int seq_len, float scale) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < seq_len) {
        const float* q_head = q + head_idx * head_dim;
        const float* k_head = k_cache + t * dim + head_idx * head_dim;
        float sum = 0.0f;
        for (int d = 0; d < head_dim; d++) sum += q_head[d] * k_head[d];
        scores[t] = sum * scale;
    }
}

__global__ void softmax_kernel(float* x, int size) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    float local_max = -1e30f, local_sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) { if (x[i] > local_max) local_max = x[i]; }
    smem[tid] = local_max; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) { if (tid < s && smem[tid+s] > smem[tid]) smem[tid] = smem[tid+s]; __syncthreads(); }
    float max_val = smem[0];
    for (int i = tid; i < size; i += blockDim.x) { x[i] = expf(x[i] - max_val); local_sum += x[i]; }
    smem[tid] = local_sum; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) { if (tid < s) smem[tid] += smem[tid+s]; __syncthreads(); }
    float total = smem[0];
    for (int i = tid; i < size; i += blockDim.x) x[i] /= total;
}

__global__ void score_v_kernel(float* out, const float* scores, const float* v_cache, int head_idx, int head_dim, int dim, int seq_len) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < head_dim) {
        float sum = 0.0f;
        for (int t = 0; t < seq_len; t++) sum += scores[t] * v_cache[t * dim + head_idx * head_dim + d];
        out[head_idx * head_dim + d] = sum;
    }
}

__global__ void apply_row_scales_kernel(const int32_t* src, float* dst, const float* scales, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = (float)src[i] * scales[i];
}

__global__ void memset_float_kernel(float* data, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) data[i] = 0.0f;
}

__global__ void copy_float_kernel(float* dst, const float* src, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) dst[i] = src[i];
}

// ============================================================
// Estruturas (buffers de estado em float, pesos INT8/half)
// ============================================================
struct Int8Layer {
    int8_t *d_wq, *d_wk, *d_wv, *d_wo;
    int8_t *d_w1, *d_w2, *d_w3;
    float *d_scale_wq, *d_scale_wk, *d_scale_wv, *d_scale_wo;
    float *d_scale_w1, *d_scale_w2, *d_scale_w3;
    half *d_norm_attn, *d_norm_ffn;       // pesos RMSNorm em half (lidos pelo kernel)
};

struct Int8Model {
    half *d_token_embedding, *d_norm_final, *d_lm_head;
    Int8Layer* layers;
    int n_layers;
};

struct Int8RunState {
    float *d_x, *d_xb, *d_q, *d_k, *d_v, *d_attn_out;
    float *d_ffn_g, *d_ffn_u, *d_ffn_out;
    float *d_logits;
    float *d_k_cache, *d_v_cache;
    float *d_rope_cos, *d_rope_sin;
    float* d_scores;
    int32_t* d_tmp_int32;
    cublasHandle_t cublas;
};

// ============================================================
// Helpers
// ============================================================
float* cuda_alloc_float(size_t n) { float* d; CUDA_CHECK(cudaMalloc(&d, n * sizeof(float))); return d; }
int8_t* cuda_alloc_int8(size_t n) { int8_t* d; CUDA_CHECK(cudaMalloc(&d, n * sizeof(int8_t))); return d; }
int32_t* cuda_alloc_int32(size_t n) { int32_t* d; CUDA_CHECK(cudaMalloc(&d, n * sizeof(int32_t))); return d; }
half* cuda_alloc_half(size_t n) { half* d; CUDA_CHECK(cudaMalloc(&d, n * sizeof(half))); return d; }

void cuda_copy_f32_to_f16(half* d, const float* h, size_t count) {
    half* tmp = (half*)malloc(count * sizeof(half));
    for (size_t i = 0; i < count; i++) tmp[i] = __float2half(h[i]);
    CUDA_CHECK(cudaMemcpy(d, tmp, count * sizeof(half), cudaMemcpyHostToDevice)); free(tmp);
}

void cuda_copy_f32_to_f32(float* d, const float* h, size_t count) {
    CUDA_CHECK(cudaMemcpy(d, h, count * sizeof(float), cudaMemcpyHostToDevice));
}

// GEMM mista: pesos INT8, ativação FP16 → saída float
void cublas_igemm_mixed(cublasHandle_t handle, const int8_t* W, const float* x, float* y,
                        int M, int K, const float* w_scales, int32_t* d_tmp) {
    // Converter ativação para half no device (a GEMM exige half como entrada)
    half* d_x_half;
    CUDA_CHECK(cudaMalloc(&d_x_half, K * sizeof(half)));
    dim3 block(256), grid((K+255)/256);
    auto to_half = [] __global__ (const float* src, half* dst, int N) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N) dst[i] = __float2half(src[i]);
    };
    to_half<<<grid, block>>>(x, d_x_half, K);

    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        M, 1, K,
        &alpha,
        W, CUDA_R_8I, M,
        d_x_half, CUDA_R_16F, K,
        &beta,
        d_tmp, CUDA_R_32I, M,
        CUDA_R_32I,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    int blocks = (M + 255) / 256;
    apply_row_scales_kernel<<<blocks, 256>>>(d_tmp, y, w_scales, M);
    CUDA_CHECK(cudaFree(d_x_half));
}

// GEMM FP16 para o LM Head
void cublas_hgemv(cublasHandle_t handle, const half* W, const float* x, float* y, int M, int K) {
    half* d_x_half;
    CUDA_CHECK(cudaMalloc(&d_x_half, K * sizeof(half)));
    dim3 block(256), grid((K+255)/256);
    auto to_half = [] __global__ (const float* src, half* dst, int N) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N) dst[i] = __float2half(src[i]);
    };
    to_half<<<grid, block>>>(x, d_x_half, K);

    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, 1, K,
        &alpha, W, CUDA_R_16F, K, d_x_half, CUDA_R_16F, K, &beta, y, CUDA_R_32F, M,
        CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CUDA_CHECK(cudaFree(d_x_half));
}

// Quantização de pesos (row-wise)
void quantize_weight_matrix_host(const float* h_data, int rows, int cols,
                                 std::vector<int8_t>& h_w, std::vector<float>& h_scales) {
    h_w.resize(rows * cols);
    h_scales.resize(rows);
    for (int r = 0; r < rows; r++) {
        float amax = 0.0f;
        for (int c = 0; c < cols; c++)
            amax = std::max(amax, std::abs(h_data[r * cols + c]));
        float scale = amax / 127.0f;
        if (scale < 1e-10f) scale = 1e-10f;
        h_scales[r] = scale;
        float inv = 1.0f / scale;
        for (int c = 0; c < cols; c++) {
            float v = roundf(h_data[r * cols + c] * inv);
            v = fmaxf(-127.0f, fminf(127.0f, v));
            h_w[r * cols + c] = (int8_t)v;
        }
    }
}

void copy_int8_and_scales_to_device(int8_t* d_w, const std::vector<int8_t>& h_w,
                                    float* d_scales, const std::vector<float>& h_scales) {
    CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), h_w.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, h_scales.data(), h_scales.size() * sizeof(float), cudaMemcpyHostToDevice));
}

// Carregamento do modelo
void int8_load_model(Int8Model& im, const Model& m) {
    im.n_layers = m.n_layers;

    im.d_token_embedding = cuda_alloc_half(m.vocab_size * m.dim);
    cuda_copy_f32_to_f16(im.d_token_embedding, m.token_embedding.data.data(), m.vocab_size * m.dim);
    im.d_norm_final = cuda_alloc_half(m.dim);
    cuda_copy_f32_to_f16(im.d_norm_final, m.norm_final.data.data(), m.dim);
    im.d_lm_head = cuda_alloc_half(m.vocab_size * m.dim);
    cuda_copy_f32_to_f16(im.d_lm_head, m.lm_head.data.data(), m.vocab_size * m.dim);

    im.layers = new Int8Layer[m.n_layers];
    for (int i = 0; i < m.n_layers; i++) {
        auto& l = m.layers[i];
        auto& cl = im.layers[i];

        std::vector<int8_t> h_int8;
        std::vector<float> h_scales;

        cl.d_wq = cuda_alloc_int8(m.dim * m.dim);
        cl.d_scale_wq = cuda_alloc_float(m.dim);
        quantize_weight_matrix_host(l.wq.data.data(), m.dim, m.dim, h_int8, h_scales);
        copy_int8_and_scales_to_device(cl.d_wq, h_int8, cl.d_scale_wq, h_scales);

        cl.d_wk = cuda_alloc_int8(m.dim * m.dim);
        cl.d_scale_wk = cuda_alloc_float(m.dim);
        quantize_weight_matrix_host(l.wk.data.data(), m.dim, m.dim, h_int8, h_scales);
        copy_int8_and_scales_to_device(cl.d_wk, h_int8, cl.d_scale_wk, h_scales);

        cl.d_wv = cuda_alloc_int8(m.dim * m.dim);
        cl.d_scale_wv = cuda_alloc_float(m.dim);
        quantize_weight_matrix_host(l.wv.data.data(), m.dim, m.dim, h_int8, h_scales);
        copy_int8_and_scales_to_device(cl.d_wv, h_int8, cl.d_scale_wv, h_scales);

        cl.d_wo = cuda_alloc_int8(m.dim * m.dim);
        cl.d_scale_wo = cuda_alloc_float(m.dim);
        quantize_weight_matrix_host(l.wo.data.data(), m.dim, m.dim, h_int8, h_scales);
        copy_int8_and_scales_to_device(cl.d_wo, h_int8, cl.d_scale_wo, h_scales);

        cl.d_w1 = cuda_alloc_int8(m.dim * m.hidden_dim);
        cl.d_scale_w1 = cuda_alloc_float(m.hidden_dim);
        quantize_weight_matrix_host(l.w1.data.data(), m.dim, m.hidden_dim, h_int8, h_scales);
        copy_int8_and_scales_to_device(cl.d_w1, h_int8, cl.d_scale_w1, h_scales);

        cl.d_w2 = cuda_alloc_int8(m.hidden_dim * m.dim);
        cl.d_scale_w2 = cuda_alloc_float(m.dim);
        quantize_weight_matrix_host(l.w2.data.data(), m.hidden_dim, m.dim, h_int8, h_scales);
        copy_int8_and_scales_to_device(cl.d_w2, h_int8, cl.d_scale_w2, h_scales);

        cl.d_w3 = cuda_alloc_int8(m.dim * m.hidden_dim);
        cl.d_scale_w3 = cuda_alloc_float(m.hidden_dim);
        quantize_weight_matrix_host(l.w3.data.data(), m.dim, m.hidden_dim, h_int8, h_scales);
        copy_int8_and_scales_to_device(cl.d_w3, h_int8, cl.d_scale_w3, h_scales);

        cl.d_norm_attn = cuda_alloc_half(m.dim);
        cuda_copy_f32_to_f16(cl.d_norm_attn, l.norm_attn.data.data(), m.dim);
        cl.d_norm_ffn = cuda_alloc_half(m.dim);
        cuda_copy_f32_to_f16(cl.d_norm_ffn, l.norm_ffn.data.data(), m.dim);
    }
    printf("[INT8 Mixed] Modelo carregado.\n");
}

void int8_init_run_state(Int8RunState& is, const Model& m) {
    is.d_x = cuda_alloc_float(m.dim); is.d_xb = cuda_alloc_float(m.dim);
    is.d_q = cuda_alloc_float(m.dim); is.d_k = cuda_alloc_float(m.dim); is.d_v = cuda_alloc_float(m.dim);
    is.d_attn_out = cuda_alloc_float(m.dim);
    is.d_ffn_g = cuda_alloc_float(m.hidden_dim); is.d_ffn_u = cuda_alloc_float(m.hidden_dim);
    is.d_ffn_out = cuda_alloc_float(m.dim);
    is.d_logits = cuda_alloc_float(m.vocab_size);

    size_t kv_elements = (size_t)m.max_seq_len * m.dim;
    is.d_k_cache = cuda_alloc_float(kv_elements);
    is.d_v_cache = cuda_alloc_float(kv_elements);

    // Zerar KV cache (float)
    int threads = 256;
    int blocks_kv = (kv_elements + threads - 1) / threads;
    memset_float_kernel<<<blocks_kv, threads>>>(is.d_k_cache, kv_elements);
    memset_float_kernel<<<blocks_kv, threads>>>(is.d_v_cache, kv_elements);

    int max_rows = std::max({m.dim, m.hidden_dim, m.vocab_size});
    is.d_tmp_int32 = cuda_alloc_int32(max_rows);

    std::vector<float> cos_buf(m.max_seq_len * m.dim), sin_buf(m.max_seq_len * m.dim);
    for (int pos = 0; pos < m.max_seq_len; pos++) {
        for (int i = 0; i < m.dim; i += 2) {
            float freq = 1.0f / powf(10000.0f, (float)i / m.dim), val = (float)pos * freq;
            cos_buf[pos * m.dim + i] = cosf(val); sin_buf[pos * m.dim + i] = sinf(val);
            cos_buf[pos * m.dim + i+1] = cosf(val); sin_buf[pos * m.dim + i+1] = sinf(val);
        }
    }
    is.d_rope_cos = cuda_alloc_float(m.max_seq_len * m.dim);
    is.d_rope_sin = cuda_alloc_float(m.max_seq_len * m.dim);
    cuda_copy_f32_to_f32(is.d_rope_cos, cos_buf.data(), m.max_seq_len * m.dim);
    cuda_copy_f32_to_f32(is.d_rope_sin, sin_buf.data(), m.max_seq_len * m.dim);
    is.d_scores = cuda_alloc_float(m.max_seq_len);
    CUBLAS_CHECK(cublasCreate(&is.cublas));
    printf("[INT8 Mixed] RunState inicializado.\n");
}

void int8_forward_fp16(Int8RunState& is, const Int8Model& im, const Model& m, int token, int pos) {
    int dim = m.dim, hidden_dim = m.hidden_dim, n_heads = m.n_heads, head_dim = dim / n_heads;
    int seq_len = pos + 1, threads = 256;

    embedding_kernel<<<(dim+threads-1)/threads, threads>>>(is.d_x, im.d_token_embedding, token, dim);

    for (int l = 0; l < m.n_layers; l++) {
        auto& cl = im.layers[l];

        rmsnorm_kernel<<<1, threads, threads*sizeof(float)>>>(is.d_xb, is.d_x, cl.d_norm_attn, dim, 1e-5f);
        cublas_igemm_mixed(is.cublas, cl.d_wq, is.d_xb, is.d_q, dim, dim, cl.d_scale_wq, is.d_tmp_int32);
        cublas_igemm_mixed(is.cublas, cl.d_wk, is.d_xb, is.d_k, dim, dim, cl.d_scale_wk, is.d_tmp_int32);
        cublas_igemm_mixed(is.cublas, cl.d_wv, is.d_xb, is.d_v, dim, dim, cl.d_scale_wv, is.d_tmp_int32);

        int rope_blocks = (dim/2 + threads - 1) / threads;
        float* cos_ptr = is.d_rope_cos + pos * dim;
        float* sin_ptr = is.d_rope_sin + pos * dim;
        rope_kernel<<<rope_blocks, threads>>>(is.d_q, is.d_k, cos_ptr, sin_ptr, pos, dim);

        // Copiar K, V para o cache (float)
        int copy_blocks = (dim + threads - 1) / threads;
        copy_float_kernel<<<copy_blocks, threads>>>(is.d_k_cache + pos * dim, is.d_k, dim);
        copy_float_kernel<<<copy_blocks, threads>>>(is.d_v_cache + pos * dim, is.d_v, dim);

        // Zerar d_attn_out (float)
        int memset_blocks = (dim + threads - 1) / threads;
        memset_float_kernel<<<memset_blocks, threads>>>(is.d_attn_out, dim);

        float scale = 1.0f / sqrtf((float)head_dim);
        for (int h = 0; h < n_heads; h++) {
            int score_blocks = (seq_len + threads - 1) / threads;
            attn_scores_kernel<<<score_blocks, threads>>>(is.d_scores, is.d_q, is.d_k_cache, h, head_dim, dim, seq_len, scale);
            softmax_kernel<<<1, threads, threads*sizeof(float)>>>(is.d_scores, seq_len);
            int sv_blocks = (head_dim + threads - 1) / threads;
            score_v_kernel<<<sv_blocks, threads>>>(is.d_attn_out, is.d_scores, is.d_v_cache, h, head_dim, dim, seq_len);
        }
        cublas_igemm_mixed(is.cublas, cl.d_wo, is.d_attn_out, is.d_xb, dim, dim, cl.d_scale_wo, is.d_tmp_int32);
        residual_kernel<<<(dim+threads-1)/threads, threads>>>(is.d_x, is.d_xb, dim);

        rmsnorm_kernel<<<1, threads, threads*sizeof(float)>>>(is.d_xb, is.d_x, cl.d_norm_ffn, dim, 1e-5f);
        cublas_igemm_mixed(is.cublas, cl.d_w1, is.d_xb, is.d_ffn_g, hidden_dim, dim, cl.d_scale_w1, is.d_tmp_int32);
        cublas_igemm_mixed(is.cublas, cl.d_w3, is.d_xb, is.d_ffn_u, hidden_dim, dim, cl.d_scale_w3, is.d_tmp_int32);
        silu_mul_kernel<<<(hidden_dim+threads-1)/threads, threads>>>(is.d_ffn_g, is.d_ffn_u, hidden_dim);
        cublas_igemm_mixed(is.cublas, cl.d_w2, is.d_ffn_g, is.d_ffn_out, dim, hidden_dim, cl.d_scale_w2, is.d_tmp_int32);
        residual_kernel<<<(dim+threads-1)/threads, threads>>>(is.d_x, is.d_ffn_out, dim);
    }

    rmsnorm_kernel<<<1, threads, threads*sizeof(float)>>>(is.d_xb, is.d_x, im.d_norm_final, dim, 1e-5f);
    cublas_hgemv(is.cublas, im.d_lm_head, is.d_xb, is.d_logits, m.vocab_size, dim);
}

int main(int argc, char* argv[]) {
    if (argc < 3) { printf("Uso: %s <modelo.bin> <tokens>\n", argv[0]); return 1; }
    const char* model_path = argv[1];
    int num_tokens = atoi(argv[2]);

    Model m;
    if (!load_model_weights(m, model_path)) { fprintf(stderr, "Erro ao carregar modelo.\n"); return 1; }
    if (num_tokens > m.max_seq_len - 1) num_tokens = m.max_seq_len - 1;

    Int8Model im;
    int8_load_model(im, m);
    Int8RunState is;
    int8_init_run_state(is, m);

    int token = 1, next, generated = 0;
    std::vector<float> logits_cpu(m.vocab_size);

    // warmup
    int8_forward_fp16(is, im, m, token, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    for (int pos = 0; pos < num_tokens; pos++) {
        int8_forward_fp16(is, im, m, token, pos);
        CUDA_CHECK(cudaMemcpy(logits_cpu.data(), is.d_logits, m.vocab_size*sizeof(float), cudaMemcpyDeviceToHost));
        int max_i = 0; float max_p = logits_cpu[0];
        for (int i = 1; i < m.vocab_size; i++) if (logits_cpu[i] > max_p) { max_i = i; max_p = logits_cpu[i]; }
        next = max_i;
        generated++;
        if (generated >= num_tokens) break;
        token = next;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    double tok_s = generated / elapsed;

    printf("\n========================================\n");
    printf(" CUDA INT8+FP16 Mixed Precision\n");
    printf("========================================\n");
    printf(" Modelo: %s\n", model_path);
    printf(" Tokens: %d\n", generated);
    printf(" Tempo: %.4f s\n", elapsed);
    printf(" Throughput: %.1f tok/s\n", tok_s);
    printf("========================================\n");

    return 0;
}