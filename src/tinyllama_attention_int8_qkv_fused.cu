#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "model.h"

//////////////////////////////////////////////////////////////
// Estrutura Quantizada (COL-MAJOR)
//////////////////////////////////////////////////////////////
struct QuantizedMatrix {
    std::vector<int8_t> weight;   // [col * out_dim + row]
    std::vector<float>  scales;   // per-row
    int out_dim;
    int in_dim;
};

//////////////////////////////////////////////////////////////
// Quantização per-row → COL-MAJOR
//////////////////////////////////////////////////////////////
QuantizedMatrix quantize_per_row_col_major(
    const std::vector<float>& W,
    int out_dim,
    int in_dim)
{
    QuantizedMatrix qm;
    qm.out_dim = out_dim;
    qm.in_dim  = in_dim;

    qm.weight.resize(out_dim * in_dim);
    qm.scales.resize(out_dim);

    for(int i = 0; i < out_dim; i++)
    {
        float max_v = 0.f;

        for(int j = 0; j < in_dim; j++)
            max_v = std::max(max_v,
                std::abs(W[i*in_dim + j]));

        float scale = max_v / 127.f;
        if(scale == 0.f) scale = 1e-8f;

        qm.scales[i] = scale;

        for(int j = 0; j < in_dim; j++)
        {
            int q = std::roundf(W[i*in_dim + j] / scale);
            q = std::max(-127, std::min(127, q));

            // COL-MAJOR
            qm.weight[j*out_dim + i] = (int8_t)q;
        }
    }

    return qm;
}

//////////////////////////////////////////////////////////////
// Fusão QKV (COL-MAJOR)
//////////////////////////////////////////////////////////////
QuantizedMatrix fuse_qkv(
    const QuantizedMatrix& q,
    const QuantizedMatrix& k,
    const QuantizedMatrix& v)
{
    QuantizedMatrix fused;

    fused.in_dim  = q.in_dim;
    fused.out_dim = q.out_dim + k.out_dim + v.out_dim;

    fused.weight.resize(fused.out_dim * fused.in_dim);
    fused.scales.resize(fused.out_dim);

    int out_offset = 0;

    auto copy_block = [&](const QuantizedMatrix& src)
    {
        for(int col = 0; col < src.in_dim; col++)
        {
            for(int row = 0; row < src.out_dim; row++)
            {
                int dst_row = out_offset + row;

                fused.weight[col * fused.out_dim + dst_row] =
                    src.weight[col * src.out_dim + row];
            }
        }

        for(int i = 0; i < src.out_dim; i++)
            fused.scales[out_offset + i] = src.scales[i];

        out_offset += src.out_dim;
    };

    copy_block(q);
    copy_block(k);
    copy_block(v);

    return fused;
}

//////////////////////////////////////////////////////////////
// MAIN
//////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    std::cout << "--- [QKV FUSED INT8 GPU] ---\n";

    if(argc < 2) {
        std::cout << "Uso: ./tinyllama_attention model.safetensors\n";
        return 0;
    }

    ////////////////////////////////////////////////////////////
    // Config modelo (TinyLlama 1.1B)
    ////////////////////////////////////////////////////////////

    Model m;
    m.dim = 2048;
    m.hidden_dim = 5632;
    m.n_layers = 22;
    m.n_heads = 32;
    m.vocab_size = 32000;
    m.max_seq_len = 2048;

    m.layers.resize(m.n_layers);

    load_safetensors_improved(argv[1], m);

    ////////////////////////////////////////////////////////////
    // Usar layer 0 para experimento
    ////////////////////////////////////////////////////////////

    auto& layer = m.layers[0];

    int q_dim = m.dim;
    int k_dim = m.dim;
    int v_dim = m.dim;

    ////////////////////////////////////////////////////////////
    // Quantizar Wq, Wk, Wv
    ////////////////////////////////////////////////////////////

    auto wq_int8 = quantize_per_row_col_major(
        layer.wq.data, q_dim, m.dim);

    auto wk_int8 = quantize_per_row_col_major(
        layer.wk.data, k_dim, m.dim);

    auto wv_int8 = quantize_per_row_col_major(
        layer.wv.data, v_dim, m.dim);

    ////////////////////////////////////////////////////////////
    // Fusão
    ////////////////////////////////////////////////////////////

    auto wqkv_int8 = fuse_qkv(
        wq_int8, wk_int8, wv_int8);

    ////////////////////////////////////////////////////////////
    // Input token + RMSNorm
    ////////////////////////////////////////////////////////////

    int token = 15043;

    std::vector<float> x(m.dim);

    for(int i=0;i<m.dim;i++)
        x[i] = m.token_embedding.data[token*m.dim + i];

    float sum=0;
    for(int i=0;i<m.dim;i++)
        sum += x[i]*x[i];

    float rms = 1.0f / sqrtf(sum/m.dim + 1e-5f);

    std::vector<float> x_norm(m.dim);

    for(int i=0;i<m.dim;i++)
        x_norm[i] = x[i] * rms *
                    layer.norm_attn.data[i];

    ////////////////////////////////////////////////////////////
    // Quantizar ativação
    ////////////////////////////////////////////////////////////

    float max_x = 0.f;
    for(float v : x_norm)
        max_x = std::max(max_x, std::abs(v));

    float scale_x = max_x / 127.f;
    if(scale_x == 0.f) scale_x = 1e-8f;

    std::vector<int8_t> x_int8(m.dim);

    for(int i=0;i<m.dim;i++)
    {
        int q = std::roundf(x_norm[i] / scale_x);
        q = std::max(-127, std::min(127, q));
        x_int8[i] = (int8_t)q;
    }

    ////////////////////////////////////////////////////////////
    // GPU
    ////////////////////////////////////////////////////////////

    cublasHandle_t handle;
    cublasCreate(&handle);

    int8_t *d_WQKV, *d_X;
    int32_t *d_QKVi32;

    cudaMalloc(&d_WQKV,
        wqkv_int8.out_dim * wqkv_int8.in_dim * sizeof(int8_t));

    cudaMalloc(&d_X, m.dim * sizeof(int8_t));

    cudaMalloc(&d_QKVi32,
        wqkv_int8.out_dim * sizeof(int32_t));

    cudaMemcpy(d_WQKV,
        wqkv_int8.weight.data(),
        wqkv_int8.out_dim * wqkv_int8.in_dim * sizeof(int8_t),
        cudaMemcpyHostToDevice);

    cudaMemcpy(d_X,
        x_int8.data(),
        m.dim * sizeof(int8_t),
        cudaMemcpyHostToDevice);

    int32_t alpha = 1;
    int32_t beta  = 0;

    ////////////////////////////////////////////////////////////
    // GEMM FUSED
    ////////////////////////////////////////////////////////////

    cublasGemmEx(handle,
                 CUBLAS_OP_N,
                 CUBLAS_OP_N,
                 wqkv_int8.out_dim,
                 1,
                 wqkv_int8.in_dim,
                 &alpha,
                 d_WQKV, CUDA_R_8I, wqkv_int8.out_dim,
                 d_X,    CUDA_R_8I, wqkv_int8.in_dim,
                 &beta,
                 d_QKVi32, CUDA_R_32I, wqkv_int8.out_dim,
                 CUBLAS_COMPUTE_32I,
                 CUBLAS_GEMM_DEFAULT);

    cudaDeviceSynchronize();

    ////////////////////////////////////////////////////////////
    // Copiar resultado
    ////////////////////////////////////////////////////////////

    std::vector<int32_t> qkv_i32(wqkv_int8.out_dim);

    cudaMemcpy(qkv_i32.data(),
               d_QKVi32,
               wqkv_int8.out_dim*sizeof(int32_t),
               cudaMemcpyDeviceToHost);

    ////////////////////////////////////////////////////////////
    // Split
    ////////////////////////////////////////////////////////////

    int32_t* q_i32 = qkv_i32.data();
    int32_t* k_i32 = qkv_i32.data() + q_dim;
    int32_t* v_i32 = qkv_i32.data() + q_dim + k_dim;

    ////////////////////////////////////////////////////////////
    // Dequant + Debug
    ////////////////////////////////////////////////////////////

    float q0 =
        q_i32[0] *
        wqkv_int8.scales[0] *
        scale_x;

    float k0 =
        k_i32[0] *
        wqkv_int8.scales[q_dim] *
        scale_x;

    float v0 =
        v_i32[0] *
        wqkv_int8.scales[q_dim + k_dim] *
        scale_x;

    std::cout << "[DEBUG] Q[0]: " << q0 << "\n";
    std::cout << "[DEBUG] K[0]: " << k0 << "\n";
    std::cout << "[DEBUG] V[0]: " << v0 << "\n";

    ////////////////////////////////////////////////////////////
    // Cleanup
    ////////////////////////////////////////////////////////////

    cudaFree(d_WQKV);
    cudaFree(d_X);
    cudaFree(d_QKVi32);
    cublasDestroy(handle);

    return 0;
}