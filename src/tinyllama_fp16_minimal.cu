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
    std::vector<int8_t> weight;
    std::vector<float> scales;
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
    qm.in_dim = in_dim;
    qm.weight.resize(out_dim * in_dim);
    qm.scales.resize(out_dim);

    for(int i=0;i<out_dim;i++)
    {
        float max_v = 0.f;

        for(int j=0;j<in_dim;j++)
            max_v = std::max(max_v,
                std::abs(W[i*in_dim + j]));

        float scale = max_v / 127.f;
        if(scale == 0.f) scale = 1e-8f;

        qm.scales[i] = scale;

        for(int j=0;j<in_dim;j++)
        {
            int q = std::roundf(W[i*in_dim + j] / scale);
            q = std::max(-127, std::min(127, q));
            qm.weight[j*out_dim + i] = (int8_t)q;
        }
    }

    return qm;
}

//////////////////////////////////////////////////////////////
// MAIN
//////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    std::cout << "--- [FFN 100% INT8 GPU] ---\n";

    if(argc < 2) {
        std::cout << "Uso: ./tinyllama_gpu model.safetensors\n";
        return 0;
    }

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
    // ✅ Pré‑quantizar W1, W3, W2
    ////////////////////////////////////////////////////////////

    auto w1_int8 = quantize_per_row_col_major(
        m.layers[0].w1.data,
        m.hidden_dim,
        m.dim);

    auto w3_int8 = quantize_per_row_col_major(
        m.layers[0].w3.data,
        m.hidden_dim,
        m.dim);

    auto w2_int8 = quantize_per_row_col_major(
        m.layers[0].w2.data,
        m.dim,
        m.hidden_dim);

    ////////////////////////////////////////////////////////////
    // Input normalizado
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
                    m.layers[0].norm_ffn.data[i];

    ////////////////////////////////////////////////////////////
    // Quantizar ativação inicial
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
    // GPU setup
    ////////////////////////////////////////////////////////////

    cublasHandle_t handle;
    cublasCreate(&handle);

    int8_t *d_W1,*d_W3,*d_W2,*d_X,*d_ACT;
    int32_t *d_W1i32,*d_W3i32,*d_W2i32;

    cudaMalloc(&d_W1, m.hidden_dim*m.dim*sizeof(int8_t));
    cudaMalloc(&d_W3, m.hidden_dim*m.dim*sizeof(int8_t));
    cudaMalloc(&d_W2, m.dim*m.hidden_dim*sizeof(int8_t));
    cudaMalloc(&d_X, m.dim*sizeof(int8_t));
    cudaMalloc(&d_ACT, m.hidden_dim*sizeof(int8_t));

    cudaMalloc(&d_W1i32, m.hidden_dim*sizeof(int32_t));
    cudaMalloc(&d_W3i32, m.hidden_dim*sizeof(int32_t));
    cudaMalloc(&d_W2i32, m.dim*sizeof(int32_t));

    cudaMemcpy(d_W1, w1_int8.weight.data(),
               m.hidden_dim*m.dim*sizeof(int8_t),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_W3, w3_int8.weight.data(),
               m.hidden_dim*m.dim*sizeof(int8_t),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_W2, w2_int8.weight.data(),
               m.dim*m.hidden_dim*sizeof(int8_t),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_X, x_int8.data(),
               m.dim*sizeof(int8_t),
               cudaMemcpyHostToDevice);

    int32_t alpha = 1;
    int32_t beta  = 0;

    ////////////////////////////////////////////////////////////
    // W1 INT8
    ////////////////////////////////////////////////////////////

    cublasGemmEx(handle,
                 CUBLAS_OP_N,CUBLAS_OP_N,
                 m.hidden_dim,1,m.dim,
                 &alpha,
                 d_W1,CUDA_R_8I,m.hidden_dim,
                 d_X,CUDA_R_8I,m.dim,
                 &beta,
                 d_W1i32,CUDA_R_32I,m.hidden_dim,
                 CUBLAS_COMPUTE_32I,
                 CUBLAS_GEMM_DEFAULT);

    ////////////////////////////////////////////////////////////
    // W3 INT8
    ////////////////////////////////////////////////////////////

    cublasGemmEx(handle,
                 CUBLAS_OP_N,CUBLAS_OP_N,
                 m.hidden_dim,1,m.dim,
                 &alpha,
                 d_W3,CUDA_R_8I,m.hidden_dim,
                 d_X,CUDA_R_8I,m.dim,
                 &beta,
                 d_W3i32,CUDA_R_32I,m.hidden_dim,
                 CUBLAS_COMPUTE_32I,
                 CUBLAS_GEMM_DEFAULT);

    cudaDeviceSynchronize();

    ////////////////////////////////////////////////////////////
    // Copiar e aplicar escala
    ////////////////////////////////////////////////////////////

    std::vector<int32_t> w1_i32(m.hidden_dim);
    std::vector<int32_t> w3_i32(m.hidden_dim);

    cudaMemcpy(w1_i32.data(), d_W1i32,
               m.hidden_dim*sizeof(int32_t),
               cudaMemcpyDeviceToHost);

    cudaMemcpy(w3_i32.data(), d_W3i32,
               m.hidden_dim*sizeof(int32_t),
               cudaMemcpyDeviceToHost);

    std::vector<float> act(m.hidden_dim);

    for(int i=0;i<m.hidden_dim;i++)
    {
        float w1f =
            w1_i32[i] *
            w1_int8.scales[i] *
            scale_x;

        float w3f =
            w3_i32[i] *
            w3_int8.scales[i] *
            scale_x;

        float silu =
            w1f / (1.0f + std::exp(-w1f));

        act[i] = silu * w3f;
    }

    ////////////////////////////////////////////////////////////
    // Quantizar ativação intermediária
    ////////////////////////////////////////////////////////////

    float max_act=0;
    for(float v:act)
        max_act=std::max(max_act,std::abs(v));

    float scale_act=max_act/127.f;
    if(scale_act==0) scale_act=1e-8f;

    std::vector<int8_t> act_int8(m.hidden_dim);

    for(int i=0;i<m.hidden_dim;i++)
    {
        int q=std::roundf(act[i]/scale_act);
        q=std::max(-127,std::min(127,q));
        act_int8[i]=(int8_t)q;
    }

    cudaMemcpy(d_ACT, act_int8.data(),
               m.hidden_dim*sizeof(int8_t),
               cudaMemcpyHostToDevice);

    ////////////////////////////////////////////////////////////
    // W2 INT8
    ////////////////////////////////////////////////////////////

    cublasGemmEx(handle,
                 CUBLAS_OP_N,CUBLAS_OP_N,
                 m.dim,1,m.hidden_dim,
                 &alpha,
                 d_W2,CUDA_R_8I,m.dim,
                 d_ACT,CUDA_R_8I,m.hidden_dim,
                 &beta,
                 d_W2i32,CUDA_R_32I,m.dim,
                 CUBLAS_COMPUTE_32I,
                 CUBLAS_GEMM_DEFAULT);

    cudaDeviceSynchronize();

    std::vector<int32_t> w2_i32(m.dim);

    cudaMemcpy(w2_i32.data(), d_W2i32,
               m.dim*sizeof(int32_t),
               cudaMemcpyDeviceToHost);

    float w20 =
        w2_i32[0] *
        w2_int8.scales[0] *
        scale_act;

    std::cout<<"[DEBUG] w2[0] INT8 GPU: "<<w20<<std::endl;

    cublasDestroy(handle);

    return 0;
}