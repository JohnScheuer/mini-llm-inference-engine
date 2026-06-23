#include "kernels/mlp_int8.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace kernels {

////////////////////////////////////////////////////////////////
// Constantes internas
////////////////////////////////////////////////////////////////
static constexpr int THREADS = 256;

////////////////////////////////////////////////////////////////
// Macro de verificação (ativa apenas em debug)
////////////////////////////////////////////////////////////////
#ifndef NDEBUG
#define KERNEL_CHECK() do {                                      \
    cudaError_t err = cudaGetLastError();                        \
    if (err != cudaSuccess) {                                    \
        printf("CUDA kernel error: %s at %s:%d\n",               \
               cudaGetErrorString(err), __FILE__, __LINE__);     \
        exit(1);                                                 \
    }                                                            \
} while(0)
#else
#define KERNEL_CHECK() ((void)0)
#endif

////////////////////////////////////////////////////////////////
// KERNEL: Quantize activation FP16 → INT8
//
// Layout: column-major [dim, batch]
// Cada bloco processa uma coluna (um item do batch)
// Redução em shared memory para achar max absoluto
////////////////////////////////////////////////////////////////
__global__ void quantize_activation_kernel(
    const half* __restrict__ in,
    int8_t* __restrict__ out,
    float* __restrict__ scale,
    int dim,
    int batch_size)
{
    int col = blockIdx.x;
    int tid = threadIdx.x;

    if (col >= batch_size) return;

    const half* col_ptr = in + col * dim;

    __shared__ float red[THREADS];

    float local_max = 0.f;
    for (int i = tid; i < dim; i += THREADS) {
        float v = fabsf(__half2float(col_ptr[i]));
        if (v > local_max) local_max = v;
    }

    red[tid] = local_max;
    __syncthreads();

    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            red[tid] = fmaxf(red[tid], red[tid + s]);
        }
        __syncthreads();
    }

    float s_val = red[0] > 0.f ? red[0] / 127.f : 1.f;

    if (tid == 0) {
        scale[col] = s_val;
    }

    __syncthreads();

    for (int i = tid; i < dim; i += THREADS) {
        float v = __half2float(col_ptr[i]) / s_val;
        int iv = static_cast<int>(roundf(v));
        iv = iv < -127 ? -127 : (iv > 127 ? 127 : iv);
        out[col * dim + i] = static_cast<int8_t>(iv);
    }
}

////////////////////////////////////////////////////////////////
// KERNEL: Dequantize INT32 → FP16
//
// Layout: column-major [m, batch]
// Cada thread processa um elemento
// scale_final = w_scale[row] * x_scale[col]
////////////////////////////////////////////////////////////////
__global__ void dequantize_kernel(
    const int32_t* __restrict__ in,
    const float* __restrict__ w_scale,
    const float* __restrict__ x_scale,
    half* __restrict__ out,
    int m,
    int batch_size)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y;

    if (row >= m || col >= batch_size) return;

    int idx = row + col * m;
    float s = w_scale[row] * x_scale[col];
    out[idx] = __float2half(static_cast<float>(in[idx]) * s);
}

////////////////////////////////////////////////////////////////
// KERNEL: SiLU Gate (SwiGLU)
//
// w1[i] = silu(w1[i]) * w3[i]
// silu(x) = x * sigmoid(x)
// sigmoid(x) = 1 / (1 + exp(-x))
//
// Modifica w1 in-place
////////////////////////////////////////////////////////////////
__global__ void silu_gate_kernel(
    half* __restrict__ w1,
    const half* __restrict__ w3,
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total) return;

    float a = __half2float(w1[idx]);
    float b = __half2float(w3[idx]);
    float sigmoid = 1.f / (1.f + expf(-a));
    float silu = a * sigmoid;
    w1[idx] = __float2half(silu * b);
}

////////////////////////////////////////////////////////////////
// WRAPPERS PÚBLICOS
////////////////////////////////////////////////////////////////

void launch_quantize_activation(
    const half* in,
    int8_t* out,
    float* scale,
    int dim,
    int batch_size,
    cudaStream_t stream)
{
    quantize_activation_kernel<<<batch_size, THREADS, 0, stream>>>(
        in, out, scale, dim, batch_size);
    KERNEL_CHECK();
}

void launch_dequantize(
    const int32_t* in,
    const float* w_scale,
    const float* x_scale,
    half* out,
    int m,
    int batch_size,
    cudaStream_t stream)
{
    dim3 grid((m + THREADS - 1) / THREADS, batch_size);
    dequantize_kernel<<<grid, THREADS, 0, stream>>>(
        in, w_scale, x_scale, out, m, batch_size);
    KERNEL_CHECK();
}

void launch_silu_gate(
    half* w1,
    const half* w3,
    int hidden_dim,
    int batch_size,
    cudaStream_t stream)
{
    int total = hidden_dim * batch_size;
    int blocks = (total + THREADS - 1) / THREADS;
    silu_gate_kernel<<<blocks, THREADS, 0, stream>>>(
        w1, w3, total);
    KERNEL_CHECK();
}

} // namespace kernels