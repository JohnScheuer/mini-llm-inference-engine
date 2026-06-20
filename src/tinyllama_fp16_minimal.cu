#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "model.h"

#define BATCH 32

//////////////////////////////////////////////////////////////
// Estrutura Quantizada
//////////////////////////////////////////////////////////////
struct QuantizedMatrix {
    std::vector<int8_t> weight;  // COL-MAJOR
    std::vector<float> scales;
    int out_dim;
    int in_dim;
};

//////////////////////////////////////////////////////////////
// Quantização per-row armazenando COL-MAJOR
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
// MAIN
//////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    std::cout << "--- [INT8 GEMM VALIDATION] ---\n";

    if(argc < 2) {
        std::cout << "Uso: ./tinyllama_tc model.safetensors\n";
        return 0;
    }

    // Inicialização completa
    Model m;
    m.dim = 2048;
    m.hidden_dim = 5632;
    m.n_layers = 22;
    m.n_heads = 32;
    m.vocab_size = 32000;
    m.max_seq_len = 2048;

    m.layers.resize(m.n_layers);

    load_safetensors_improved(argv[1], m);

    cublasHandle_t handle;
    cublasCreate(&handle);

    ////////////////////////////////////////////////////////////
    // Pré‑quantizar Wq col-major
    ////////////////////////////////////////////////////////////

    auto wq = quantize_per_row_col_major(
        m.layers[0].wq.data,
        m.dim,
        m.dim);

    ////////////////////////////////////////////////////////////
    // Criar input normalizado
    ////////////////////////////////////////////////////////////

    int token = 15043;

    std::vector<float> x(m.dim);

    for(int i=0;i<m.dim;i++)
        x[i] = (float)m.token_embedding.data[token*m.dim + i];

    float sum=0;
    for(int i=0;i<m.dim;i++)
        sum += x[i]*x[i];

    float rms = 1.0f / sqrtf(sum/m.dim + 1e-5f);

    std::vector<float> x_norm(m.dim);

    for(int i=0;i<m.dim;i++)
        x_norm[i] = x[i] * rms *
                    (float)m.layers[0].norm_attn.data[i];

    ////////////////////////////////////////////////////////////
    // Quantizar ativação
    ////////////////////////////////////////////////////////////

    float max_x = 0.f;
    for(int i=0;i<m.dim;i++)
        max_x = std::max(max_x, std::abs(x_norm[i]));

    float scale_x = max_x / 127.f;
    if(scale_x == 0.f) scale_x = 1e-8f;

    std::vector<int8_t> x_int8(m.dim * BATCH);

    for(int b=0;b<BATCH;b++)
    {
        for(int i=0;i<m.dim;i++)
        {
            int q = std::roundf(x_norm[i] / scale_x);
            q = std::max(-127, std::min(127, q));
            x_int8[i + b*m.dim] = (int8_t)q; // COL-MAJOR
        }
    }

    ////////////////////////////////////////////////////////////
    // ✅ CPU COL-MAJOR REFERÊNCIA
    ////////////////////////////////////////////////////////////

    int32_t acc_cpu = 0;

    for(int j=0; j<m.dim; j++)
    {
        int8_t w = wq.weight[j*m.dim + 0]; // col-major
        int8_t xv = x_int8[j];             // primeira coluna
        acc_cpu += (int32_t)w * (int32_t)xv;
    }

    float q_cpu =
        acc_cpu *
        wq.scales[0] *
        scale_x;

    std::cout << "[DEBUG] CPU col-major q[0]: "
              << q_cpu << std::endl;

    ////////////////////////////////////////////////////////////
    // GPU
    ////////////////////////////////////////////////////////////

    int8_t* d_W;
    int8_t* d_X;
    int32_t* d_Y;

    cudaMalloc(&d_W, m.dim*m.dim*sizeof(int8_t));
    cudaMalloc(&d_X, m.dim*BATCH*sizeof(int8_t));
    cudaMalloc(&d_Y, m.dim*BATCH*sizeof(int32_t));

    cudaMemcpy(d_W,
               wq.weight.data(),
               m.dim*m.dim*sizeof(int8_t),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_X,
               x_int8.data(),
               m.dim*BATCH*sizeof(int8_t),
               cudaMemcpyHostToDevice);

    int M = m.dim;
    int N = BATCH;
    int K = m.dim;

    int32_t alpha = 1;
    int32_t beta  = 0;

    cublasGemmEx(handle,
                 CUBLAS_OP_N,
                 CUBLAS_OP_N,
                 M,
                 N,
                 K,
                 &alpha,
                 d_W,
                 CUDA_R_8I,
                 M,
                 d_X,
                 CUDA_R_8I,
                 K,
                 &beta,
                 d_Y,
                 CUDA_R_32I,
                 M,
                 CUBLAS_COMPUTE_32I,
                 CUBLAS_GEMM_DEFAULT);

    cudaDeviceSynchronize();

    std::vector<int32_t> y_int32(m.dim * BATCH);

    cudaMemcpy(y_int32.data(),
               d_Y,
               m.dim*BATCH*sizeof(int32_t),
               cudaMemcpyDeviceToHost);

    float q_gpu =
        y_int32[0] *
        wq.scales[0] *
        scale_x;

    std::cout << "[DEBUG] GPU q[0]: "
              << q_gpu << std::endl;

    cudaFree(d_W);
    cudaFree(d_X);
    cudaFree(d_Y);

    cublasDestroy(handle);

    return 0;
}