#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "model.h"

//////////////////////////////////////////////////////////////
// Estrutura Col-Major
//////////////////////////////////////////////////////////////
struct ColMajorMatrix {
    std::vector<float> weight;
    int out_dim;
    int in_dim;
};

//////////////////////////////////////////////////////////////
// Converter Row → Col Major
//////////////////////////////////////////////////////////////
ColMajorMatrix to_col_major(
    const std::vector<float>& W,
    int out_dim,
    int in_dim)
{
    ColMajorMatrix cm;
    cm.out_dim = out_dim;
    cm.in_dim = in_dim;
    cm.weight.resize(out_dim * in_dim);

    for(int i=0;i<out_dim;i++)
        for(int j=0;j<in_dim;j++)
            cm.weight[j*out_dim + i] =
                W[i*in_dim + j];

    return cm;
}

//////////////////////////////////////////////////////////////
// CPU GEMM
//////////////////////////////////////////////////////////////
void gemm_col_major(
    const ColMajorMatrix& W,
    const std::vector<float>& x,
    std::vector<float>& y)
{
    int M = W.out_dim;
    int K = W.in_dim;

    y.assign(M, 0.f);

    for(int i=0;i<M;i++)
    {
        float acc = 0.f;

        for(int j=0;j<K;j++)
            acc +=
                W.weight[j*M + i] *
                x[j];

        y[i] = acc;
    }
}

//////////////////////////////////////////////////////////////
// MAIN
//////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    std::cout << "--- [FFN 100% GPU FP32] ---\n";

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
    // Converter W1, W3, W2 camada 0 para col-major
    ////////////////////////////////////////////////////////////

    auto w1 = to_col_major(
        m.layers[0].w1.data,
        m.hidden_dim,
        m.dim);

    auto w3 = to_col_major(
        m.layers[0].w3.data,
        m.hidden_dim,
        m.dim);

    auto w2 = to_col_major(
        m.layers[0].w2.data,
        m.dim,
        m.hidden_dim);

    ////////////////////////////////////////////////////////////
    // Criar input normalizado
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
    // CPU referência
    ////////////////////////////////////////////////////////////

    std::vector<float> w1_cpu, w3_cpu;

    gemm_col_major(w1, x_norm, w1_cpu);
    gemm_col_major(w3, x_norm, w3_cpu);

    // SiLU * w3
    std::vector<float> act_cpu(m.hidden_dim);

    for(int i=0;i<m.hidden_dim;i++)
    {
        float silu = w1_cpu[i] /
                     (1.0f + std::exp(-w1_cpu[i]));
        act_cpu[i] = silu * w3_cpu[i];
    }

    std::vector<float> w2_cpu;
    gemm_col_major(w2, act_cpu, w2_cpu);

    std::cout << "[CPU] w2[0]: "
              << w2_cpu[0] << std::endl;

    ////////////////////////////////////////////////////////////
    // GPU
    ////////////////////////////////////////////////////////////

    cublasHandle_t handle;
    cublasCreate(&handle);

    float *d_W1, *d_W3, *d_W2;
    float *d_X, *d_W1out, *d_W3out, *d_ACT, *d_W2out;

    cudaMalloc(&d_W1, m.hidden_dim*m.dim*sizeof(float));
    cudaMalloc(&d_W3, m.hidden_dim*m.dim*sizeof(float));
    cudaMalloc(&d_W2, m.dim*m.hidden_dim*sizeof(float));

    cudaMalloc(&d_X, m.dim*sizeof(float));
    cudaMalloc(&d_W1out, m.hidden_dim*sizeof(float));
    cudaMalloc(&d_W3out, m.hidden_dim*sizeof(float));
    cudaMalloc(&d_ACT, m.hidden_dim*sizeof(float));
    cudaMalloc(&d_W2out, m.dim*sizeof(float));

    cudaMemcpy(d_W1, w1.weight.data(),
               m.hidden_dim*m.dim*sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_W3, w3.weight.data(),
               m.hidden_dim*m.dim*sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_W2, w2.weight.data(),
               m.dim*m.hidden_dim*sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_X, x_norm.data(),
               m.dim*sizeof(float),
               cudaMemcpyHostToDevice);

    float alpha = 1.0f;
    float beta  = 0.0f;

    // W1
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                m.hidden_dim, 1, m.dim,
                &alpha,
                d_W1, m.hidden_dim,
                d_X,  m.dim,
                &beta,
                d_W1out, m.hidden_dim);

    // W3
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                m.hidden_dim, 1, m.dim,
                &alpha,
                d_W3, m.hidden_dim,
                d_X,  m.dim,
                &beta,
                d_W3out, m.hidden_dim);

    cudaDeviceSynchronize();

    std::vector<float> w1_gpu(m.hidden_dim);
    std::vector<float> w3_gpu(m.hidden_dim);

    cudaMemcpy(w1_gpu.data(), d_W1out,
               m.hidden_dim*sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaMemcpy(w3_gpu.data(), d_W3out,
               m.hidden_dim*sizeof(float),
               cudaMemcpyDeviceToHost);

    // SiLU * w3 (CPU para simplicidade)
    std::vector<float> act_gpu(m.hidden_dim);

    for(int i=0;i<m.hidden_dim;i++)
    {
        float silu = w1_gpu[i] /
                     (1.0f + std::exp(-w1_gpu[i]));
        act_gpu[i] = silu * w3_gpu[i];
    }

    cudaMemcpy(d_ACT, act_gpu.data(),
               m.hidden_dim*sizeof(float),
               cudaMemcpyHostToDevice);

    // W2
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                m.dim, 1, m.hidden_dim,
                &alpha,
                d_W2, m.dim,
                d_ACT, m.hidden_dim,
                &beta,
                d_W2out, m.dim);

    cudaDeviceSynchronize();

    std::vector<float> w2_gpu(m.dim);

    cudaMemcpy(w2_gpu.data(),
               d_W2out,
               m.dim*sizeof(float),
               cudaMemcpyDeviceToHost);

    std::cout << "[GPU] w2[0]: "
              << w2_gpu[0] << std::endl;

    cudaFree(d_W1); cudaFree(d_W3); cudaFree(d_W2);
    cudaFree(d_X); cudaFree(d_W1out); cudaFree(d_W3out);
    cudaFree(d_ACT); cudaFree(d_W2out);

    cublasDestroy(handle);

    return 0;
}