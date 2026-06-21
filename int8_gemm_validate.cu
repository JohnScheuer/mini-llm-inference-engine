#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA err '%s' @%d\n",cudaGetErrorString(e),__LINE__); exit(1);} } while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){printf("CUBLAS err %d @%d\n",s,__LINE__); exit(1);} } while(0)
#define SYNC_CHECK(msg) do { cudaError_t e=cudaDeviceSynchronize(); if(e!=cudaSuccess){printf("SYNC err '%s' at %s @%d\n",cudaGetErrorString(e),msg,__LINE__); exit(1);} } while(0)

__global__ void dequant_kernel(float* out, const int32_t* acc,
                               const float* scale_w, float scale_act, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=n) return;
    out[i] = (float)acc[i] * scale_w[i] * scale_act;
}

int main(){
    const int m = 2048, k = 2048, n = 1;
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    std::vector<float> hW(m*k), hA(k);
    for(auto& x: hW) x = dist(rng);
    for(auto& x: hA) x = dist(rng);

    // per-row weight quant
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
    // per-tensor activation quant
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
    CUDA_CHECK(cudaMalloc(&dWq,  m*k*sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&dAq,  k*sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&dC,   m*n*sizeof(int32_t)));   // FIX
    CUDA_CHECK(cudaMalloc(&dSw,  m*sizeof(float)));        // FIX
    CUDA_CHECK(cudaMalloc(&dCfp, m*n*sizeof(float)));      // FIX
    CUDA_CHECK(cudaMemcpy(dWq,hWq.data(),m*k*sizeof(int8_t),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dAq,hAq.data(),k*sizeof(int8_t),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dSw,hSw.data(),m*sizeof(float),cudaMemcpyHostToDevice));

    // ---------- INT8 GEMM via cublasLt ----------
    cublasLtHandle_t lt;
    CUBLAS_CHECK(cublasLtCreate(&lt));
    cublasLtMatmulDesc_t opDesc;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I));
    cublasOperation_t opN = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, m, k, m));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, k, n, k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32I, m, n, m));

    cublasLtMatmulPreference_t pref;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    size_t wsSize = 4*1024*1024;
    void* ws; CUDA_CHECK(cudaMalloc(&ws, wsSize));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsSize, sizeof(wsSize)));

    cublasLtMatmulHeuristicResult_t heur[4];
    int returned = 0;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(lt, opDesc, Adesc, Bdesc, Cdesc, Cdesc, pref, 4, heur, &returned));
    printf("INT8 algos encontrados: %d\n", returned);
    if(returned==0){ printf("Nenhum algo INT8 disponivel. Abortando.\n"); return 1; }
    if(heur[0].state != CUBLAS_STATUS_SUCCESS){ printf("Heuristic[0] state=%d (invalido)\n", heur[0].state); return 1; }
    cublasLtMatmulAlgo_t algo = heur[0].algo;

    int32_t alpha=1, beta=0;
    CUBLAS_CHECK(cublasLtMatmul(lt, opDesc, &alpha, dWq, Adesc, dAq, Bdesc, &beta, dC, Cdesc, dC, Cdesc, &algo, ws, wsSize, 0));
    SYNC_CHECK("apos INT8 cublasLtMatmul");

    int threads=256, blocks=(m+threads-1)/threads;
    dequant_kernel<<<blocks,threads>>>(dCfp,dC,dSw,Sa,m);
    SYNC_CHECK("apos dequant_kernel");

    std::vector<float> hCint8(m);
    CUDA_CHECK(cudaMemcpy(hCint8.data(),dCfp,m*sizeof(float),cudaMemcpyDeviceToHost));

    // ---------- referencia FP32 ----------
    cublasHandle_t h; CUBLAS_CHECK(cublasCreate(&h));
    float *dW,*dA,*dCref;
    CUDA_CHECK(cudaMalloc(&dW,m*k*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dA,k*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dCref,m*n*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dW,hW.data(),m*k*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA,hA.data(),k*sizeof(float),cudaMemcpyHostToDevice));
    float fa=1.f, fb=0.f;
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&fa,dW,m,dA,k,&fb,dCref,m));
    SYNC_CHECK("apos cublasSgemm");
    std::vector<float> hCref(m);
    CUDA_CHECK(cudaMemcpy(hCref.data(),dCref,m*sizeof(float),cudaMemcpyDeviceToHost));

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

    // ---------- benchmark ----------
    int iters=200;
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for(int it=0;it<iters;++it){
        cublasLtMatmul(lt, opDesc, &alpha, dWq, Adesc, dAq, Bdesc, &beta, dC, Cdesc, dC, Cdesc, &algo, ws, wsSize, 0);
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
    printf("\n=== GEMM decode (n=1), effective weight-read BW ===\n");
    printf("INT8 : %.4f ms/iter | %.1f GB/s\n", ms8/iters, gb8);
    printf("FP16 : %.4f ms/iter | %.1f GB/s\n", ms16/iters, gb16);
    printf("speedup: %.2fx\n", ms16/ms8);
    return 0;
}
