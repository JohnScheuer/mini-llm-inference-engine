# Mini-LLM Inference Engine (HPC Focus)

A bare-metal Transformer inference engine built in pure C++17 and CUDA,
designed to push hardware to its limits — from x86 SIMD to NVIDIA Tensor
Cores.

## 🚀 Performance

| Backend | Model | Tokens | Throughput | Latency |
|---|---|---|---|---|
| CPU (C++17, AVX2, INT8) | stories15M | 100 | **2,730 tok/s** | 0.37 ms |
| CPU (C++17, AVX2, INT8) | stories42M | 512 | **685 tok/s** | 1.46 ms |
| CPU (C++17, AVX2, INT8) | stories110M | 512 | **246 tok/s** | 4.06 ms |
| GPU (CUDA, FP16 + Tensor Cores) | stories110M | 512 | **1,740 tok/s** | 0.57 ms |

**CPU:** AMD Ryzen 5 5600X (6 physical cores)  
**GPU:** NVIDIA GeForce RTX 2070 (8 GB, Turing)

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



### CPU Backend (C++17)
- **INT8 Quantization (W8A32):** 75% reduction in memory bandwidth.
- **AVX2/FMA Micro-kernel:** Hand‑written `vpmaddubsw` chain for s8×s8
  dot products.
- **Zero-Allocation Hot Path:** Pre‑allocated RunState buffers eliminate
  `malloc`/`free` during inference.
- **FlashAttention‑style Tiled Attention:** Online softmax with dynamic
  tile sizes (32/64), skip‑correction, and 2× unroll.
- **Pre‑computed RoPE Tables:** O(1) lookup instead of per‑token trig.
- **Redundant Quantization Elimination:** Single quantization per input
  shared across multiple projections.
- **Physical Core Pinning:** Only physical cores used to avoid SMT
  contention.
- **Adaptive OpenMP:** Parallelism activated only for matrices with
  N≥512.

### GPU Backend (CUDA)
- **FP16 Precision:** Weights and activations in half‑precision.
- **Tensor Cores (Turing):** All matrix multiplications use
  `cublasGemmEx` with `CUBLAS_COMPUTE_32F_FAST_16F`.
- **Custom CUDA Kernels:** RMSNorm, RoPE, SwiGLU, residual, attention
  scores/softmax/score×V.
- **Full Inference on GPU:** Only logits transferred to host for
  sampling.

## 📦 Project Structure
mini-llm-inference-engine/
├── src/
│ ├── tensor_int8.h # INT8 block quantization types
│ ├── model.h # Model, RunState, KVCache definitions
│ ├── matmul_blocked.h/cpp # AVX2 INT8 GEMV micro‑kernel
│ ├── layers.h/cpp # Transformer layers (CPU)
│ ├── quantize.cpp # AVX2‑vectorized dynamic quantization
│ ├── model.cpp # Weight loader (llama2.c format)
│ ├── main.cpp # Inference loop (CPU)
│ ├── cuda_llm_fp16.cu # Complete GPU backend
│ └── cuda_benchmark.cu # Synthetic CUDA kernel benchmarks
├── models/ # Pre‑trained models (llama2.c format)
├── README.md
└── Makefile / build scripts



## ⚙️ Build & Run

### CPU Backend
```bash
g++ -std=c++17 -O3 -march=native -mavx2 -mfma -fopenmp \
    src/quantize.cpp src/matmul_blocked.cpp src/layers.cpp \
    src/model.cpp src/main.cpp \
    -I./src -o mini_llm

./mini_llm models/stories15M.bin 100
GPU Backend (requires CUDA Toolkit 12.x)
Bash

nvcc -std=c++17 -O3 -arch=sm_75 -ccbin /usr/bin/g++-12 \
    -I./src \
    src/cuda_llm_fp16.cu src/model.cpp \
    -lcublas -o cuda_llm

./cuda_llm models/stories110M.bin 512
🧪 Benchmarks
All benchmarks use greedy decoding and INT8 (CPU) or FP16 (GPU)
quantization.

Model	Parameters	Layers	Dim	CPU tok/s	GPU tok/s	Speedup
stories15M	15M	6	288	2,730	—	—
stories42M	42M	8	512	685	—	—
stories110M	110M	12	768	246	1,740	7.1×
🔬 Key Optimizations
CPU: INT8 → AVX2 maddubs chain → 60% throughput gain.
CPU: Tiled Attention with online softmax → up to 10% speedup on
long sequences.
GPU: FP16 + Tensor Cores → 7.1× faster than CPU on 110M model.
GPU: cuBLAS GEMM for all projections, custom kernels for
element‑wise ops.
🚧 Ongoing Work
Fused CUDA kernels (RMSNorm+QKV, RoPE+Attention) — experimental, not
yet beating cuBLAS for small dims.
FlashAttention‑2 style Shared Memory Tiling on GPU.
Batch inference for serving scenarios.
📝 License
MIT

