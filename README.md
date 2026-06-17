# Mini-LLM Inference Engine (HPC Focus)

A bare-metal Transformer inference engine built in pure C++17 and CUDA,
designed to push hardware to its theoretical limits — from x86 SIMD to
NVIDIA Tensor Cores.

## 🚀 Performance Benchmarks

| Backend | Model | Tokens | Throughput | Latency | Speedup |
|---|---|---|---|---|---|
| CPU (C++/AVX2/INT8) | stories15M | 100 | **2,730 tok/s** | 0.37 ms | — |
| CPU (C++/AVX2/INT8) | stories42M | 512 | **685 tok/s** | 1.46 ms | — |
| CPU (C++/AVX2/INT8) | stories110M | 512 | **246 tok/s** | 4.06 ms | — |
| **GPU FP16 (cuBLAS + TC)** | **stories110M** | **512** | **1,740 tok/s** | **0.57 ms** | **7.1×** |

**CPU:** AMD Ryzen 5 5600X (6 physical cores)  
**GPU:** NVIDIA GeForce RTX 2070 (8 GB, Turing, SM 7.5)

---

## 🧠 Architecture
Token ID → Embedding Lookup
↓
┌─── × N Layers ───────────────────┐
│ RMSNorm → Q/K/V Projections │
│ → RoPE (Rotary Embeddings)│
│ → KV Cache │
│ → Multi-Head Attention │
│ → Output Projection (Wo) │
│ → + Residual │
│ RMSNorm → FFN (SwiGLU) │
│ → w1(gate), w3(up) │
│ → silu(gate) * up │
│ → w2(down) │
│ → + Residual │
└──────────────────────────────────┘
↓
RMSNorm Final → LM Head → Argmax → Next Token

---

## ⚙️ Build & Run

### CPU Backend (C++17 + AVX2 + INT8)
```bash
g++ -std=c++17 -O3 -march=native -mavx2 -mfma -fopenmp \
    src/quantize.cpp src/matmul_blocked.cpp src/layers.cpp \
    src/model.cpp src/main.cpp \
    -I./src -o mini_llm

./mini_llm models/stories15M.bin 100

GPU Backend (CUDA FP16 + Tensor Cores)
nvcc -std=c++17 -O3 -arch=sm_75 -ccbin /usr/bin/g++-12 \
    -I./src \
    src/cuda_llm_fp16.cu src/model.cpp \
    -lcublas -o cuda_llm

./cuda_llm models/stories110M.bin 512

🛠️ Optimizations
CPU
INT8 Quantization (W8A32): Block‑wise (QK=32) dynamic quantization
with AVX2 vpmaddubsw instruction chain.
FlashAttention‑style Tiled Attention: Online softmax with dynamic
tile sizes (32/64), skip‑correction, 2× unroll.
Zero‑Allocation Hot Path: Pre‑allocated RunState buffers eliminate
all malloc/free during autoregressive inference.
Pre‑computed RoPE Tables: O(1) lookup instead of per‑token trig.
Redundant Quantization Elimination: Single quantization per input
shared across multiple projections.
Physical Core Pinning: Only physical cores used to avoid SMT
contention.
Adaptive OpenMP: Parallelism activated only for matrices with
N≥512.
GPU
FP16 + Tensor Cores: All matrix multiplications via
cublasGemmEx with CUDA_R_16F and CUBLAS_COMPUTE_32F_FAST_16F.
Custom CUDA Kernels: RMSNorm, RoPE, SwiGLU, residual, attention
score/softmax/score×V.
Full Inference on GPU: Only logits transferred to host for
sampling.

📦 Project Structure
mini-llm-inference-engine/
├── src/
│   ├── tensor_int8.h           # INT8 block quantization types
│   ├── model.h                 # Model, KVCache, TransformerLayer,
│   │                           #   RunState
│   ├── model.cpp               # Weight loader (llama2.c format)
│   ├── matmul_blocked.h/cpp    # AVX2 INT8 GEMV micro‑kernel
│   ├── layers.h/cpp            # Transformer layers (CPU)
│   ├── quantize.cpp            # AVX2‑vectorized dynamic quantization
│   ├── main.cpp                # Inference loop (CPU)
│   ├── cuda_llm_fp16.cu        # Complete GPU backend (FP16)
│   ├── cuda_llm_int8_batched.cu# Experimental INT8 backend
│   └── bench_int8_gemm.cu      # Standalone INT8 GEMM benchmark
├── models/                     # Pre‑trained models (llama2.c format)
├── README.md
└── .gitignore

🔬 Experimental: INT8 Mixed Precision
We validated pure INT8 GEMM (act + weights) with Tensor Cores in isolated
benchmarks, achieving up to 3200 GFLOPS on the RTX 2070. A complete
INT8 pipeline using dynamic per‑token quantization and int32
accumulation was built and is functional:

Backend	Model	Throughput
GPU INT8 (dynamic)	stories15M	542 tok/s
GPU INT8 (dynamic)	stories110M	168 tok/s
The overhead of per‑token quantization limits batch‑1 performance. For
batch inference or larger models (>1B), pure INT8 can surpass FP16. Our
implementation is available in src/cuda_llm_int8_batched.cu and serves
as a reference for integrating quantization engines.

📝 License
MIT