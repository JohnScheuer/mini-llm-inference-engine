#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA err %s @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} } while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){printf("CUBLAS err %d @%d\n",s,__LINE__); exit(1);} } while(0)

// dequant: out[i] = acc[i] * scale_w[i] * scale_act
__global__ void dequant_kernel(float* out, const int32_t* acc,
                               const float* scale_w, float scale_act, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=n) return;
    out[i] = (float)acc[i] * scale_w[i] * scale_act;
}

int main(){
    // imita decode: n = 1
    const int m = 2048;   // out_dim
    const int k = 2048;   // in_dim
    const int n = 1;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    std::vector<float> hW(m*k), hA(k);
    for(auto& x: hW) x = dist(rng);
    for(auto& x: hA) x = dist(rng);

    // ---- per-row weight quant (host) ----
    std::vector<int8_t> hWq(m*k);
    std::vector<float>  hSw(m);
    for(int r=0;r<m;++r){
        float amax=0.f;
        for(int c=0;c<k;++c) amax=fmaxf(amax, fabsf(hW[c*m + r]));
        float s = amax>0.f? amax/127.f : 1.f;
        hSw[r]=s;
        for(int c=0;c<k;++c){
            int q=(int)roundf(hW[c*m+r]/s);
            q=std::max(-127,std::min(127,q));
            hWq[c*m+r]=(int8_t)q;
        }
    }
    // ---- per-tensor activation quant (host) ----
    std::vector<int8_t> hAq(k);
    float Sa;
    {
        float amax=0.f;
        for(int i=0;i<k;++i) amax=fmaxf(amax,fabsf(hA[i]));
        Sa = amax>0.f? amax/127.f : 1.f;
        for(int i=0;i<k;++i){
            int q=(int)roundf(hA[i]/Sa);
            q=std::max(-127,std::min(127,q));
            hAq[i]=(int8_t)q;
        }
    }

    int8_t *dWq,*dAq; int32_t *dC; float *dSw,*dCfp;
    CUDA_CHECK(cudaMalloc(&dWq,m*k));
    CUDA_CHECK(cudaMalloc(&dAq,k));
    CUDA_CHECK(cudaMalloc(&dC,m*n));
    CUDA_CHECK(cudaMalloc(&dSw,m));
    CUDA_CHECK(cudaMalloc(&dCfp,m*n));
    CUDA_CHECK(cudaMemcpy(dWq,hWq.data(),m*k,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dAq,hAq.data(),k,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dSw,hSw.data(),m*sizeof(float),cudaMemcpyHostToDevice));

    cublasHandle_t h; CUBLAS_CHECK(cublasCreate(&h));

    // ---- INT8 GEMM: C_int32 = Wq * Aq ----
    int32_t alpha=1, beta=0;
    CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N,
        m,n,k,
        &alpha,
        dWq, CUDA_R_8I, m,
        dAq, CUDA_R_8I, k,
        &beta,
        dC, CUDA_R_32I, m,
        CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT));

    int threads=256, blocks=(m+threads-1)/threads;
    dequant_kernel<<<blocks,threads>>>(dCfp,dC,dSw,Sa,m);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> hCint8(m);
    CUDA_CHECK(cudaMemcpy(hCint8.data(),dCfp,m*sizeof(float),cudaMemcpyDeviceToHost));

    // ---- referencia FP32: C_ref = W * A ----
    float *dW,*dA,*dCref;
    CUDA_CHECK(cudaMalloc(&dW,m*k*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dA,k*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dCref,m*n*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dW,hW.data(),m*k*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA,hA.data(),k*sizeof(float),cudaMemcpyHostToDevice));
    float fa=1.f, fb=0.f;
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&fa,dW,m,dA,k,&fb,dCref,m));
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> hCref(m);
    CUDA_CHECK(cudaMemcpy(hCref.data(),dCref,m*sizeof(float),cudaMemcpyDeviceToHost));

    // ---- compara ----
    float max_abs=0.f, max_rel=0.f; double sum_abs=0.0;
    for(int i=0;i<m;++i){
        float a=hCint8[i], b=hCref[i];
        float d=fabsf(a-b);
        sum_abs+=d;
        max_abs=fmaxf(max_abs,d);
        float denom=fmaxf(1e-6f,fabsf(b));
        max_rel=fmaxf(max_rel,d/denom);
    }
    printf("=== INT8 GEMM validation (m=%d k=%d n=%d) ===\n",m,k,n);
    printf("max abs err : %.6f\n", max_abs);
    printf("max rel err : %.4f %%\n", max_rel*100.f);
    printf("mean abs err: %.6f\n", sum_abs/m);
    printf("sample[0]   : int8=%.5f  fp32=%.5f\n", hCint8[0], hCref[0]);

    // ---- benchmark INT8 vs FP16 GEMM ----
    int iters=200;
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);

    cudaEventRecord(t0);
    for(int it=0;it<iters;++it){
        cublasGemmEx(h, CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&alpha,
            dWq,CUDA_R_8I,m,dAq,CUDA_R_8I,k,&beta,dC,CUDA_R_32I,m,
            CUBLAS_COMPUTE_32I,CUBLAS_GEMM_DEFAULT);
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms8; cudaEventElapsedTime(&ms8,t0,t1);

    half *dWh,*dAh,*dCh;
    CUDA_CHECK(cudaMalloc(&dWh,m*k*sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dAh,k*sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dCh,m*sizeof(half)));
    std::vector<half> hWh(m*k), hAh(k);
    for(int i=0;i<m*k;++i) hWh[i]=__float2half(hW[i]);
    for(int i=0;i<k;++i) hAh[i]=__float2half(hA[i]);
    CUDA_CHECK(cudaMemcpy(dWh,hWh.data(),m*k*sizeof(half),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dAh,hAh.data(),k*sizeof(half),cudaMemcpyHostToDevice));
    float halpha=1.f,hbeta=0.f;
    cudaEventRecord(t0);
    for(int it=0;it<iters;++it){
        cublasGemmEx(h, CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&halpha,
            dWh,CUDA_R_16F,m,dAh,CUDA_R_16F,k,&hbeta,dCh,CUDA_R_16F,m,
            CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms16; cudaEventElapsedTime(&ms16,t0,t1);

    double gb8  = (double)m*k*1.0*iters/(ms8/1000.0)/1e9;
    double gb16 = (double)m*k*2.0*iters/(ms16/1000.0)/1e9;
    printf("\n=== GEMM decode (n=1), efetive weight-read BW ===\n");
    printf("INT8 : %.4f ms/iter | %.1f GB/s\n", ms8/iters, gb8);
    printf("FP16 : %.4f ms/iter | %.1f GB/s\n", ms16/iters, gb16);
    printf("speedup: %.2fx\n", ms16/ms8);
    return 0;
}