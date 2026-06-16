# Mini-LLM Inference Engine (HPC Focus)

A bare-metal Transformer inference engine built in pure C++17, designed
to push the theoretical performance limits of x86_64 CPUs through
low-level optimization.

## 🚀 Performance Benchmarks

All benchmarks: AMD Ryzen 5 5600X (6 physical cores), INT8 W8A32, greedy decoding.

| Model | Params | Layers | Dim | Tokens | Throughput | Latency |
|---|---|---|---|---|---|---|
| stories15M | 15M | 6 | 288 | 100 | **2,411 tok/s** | 0.41 ms |
| stories42M | 42M | 8 | 512 | 512 | **645 tok/s** | 1.55 ms |
| stories42M | 42M | 8 | 512 | 1,023 | **544 tok/s** | 1.84 ms |
| stories110M | 110M | 12 | 768 | 512 | **229 tok/s** | 4.37 ms |
| stories110M | 110M | 12 | 768 | 1,023 | **203 tok/s** | 4.93 ms |

### GPU Benchmarks (RTX 2070, 8GB)

| Kernel | Latência |
|:---|:---|
| RMSNorm | 6.6 μs |
| RoPE | 5.1 μs |
| SiLU (SwiGLU) | 4.7 μs |
| **Total por camada** | **~16.4 μs** |
| **Throughput estimado** | **~10,000 tok/s** (kernels only) |

> **Nota:** Estes números são apenas dos kernels customizados (RMSNorm, RoPE, SiLU). 
> A integração completa com cuBLAS para as projeções Q/K/V/FFN está em desenvolvimento.

## 🛠️ Optimizations Implemented

### Compute
- **INT8 Quantization (W8A32):** 75% reduction in memory bandwidth
  and cache footprint via block-wise quantization (QK=32).
- **AVX2/FMA Micro-kernel:** Hand-written SIMD intrinsics for
  256-bit vectorized INT8 GEMV with `vpmaddubsw` instruction chain
  using `abs/sign` trick to handle u8×s8 type constraints.
- **2x Loop Unrolling:** Dual accumulators to exploit Instruction-Level
  Parallelism (ILP) on Zen 3 micro-architecture.

### Memory
- **Zero-Allocation Hot Path:** Pre-allocated RunState buffers
  eliminate all `malloc`/`free` calls during autoregressive inference,
  preventing cache pollution and page faults.
- **L3 Cache Residency:** Working set optimized to fit entirely within
  the 32MB L3 cache, eliminating the DRAM bandwidth bottleneck.
- **Redundant Quantization Elimination:** Input vectors shared across
  projections (Q/K/V and Gate/Up) are quantized once and reused,
  reducing dynamic quantization calls by 57%.

### Attention
- **AVX2 Dot Product:** Vectorized Q·Kᵀ and Score×V operations with
  FMA, replacing scalar loops in the attention mechanism.
- **Pre-computed RoPE Tables:** Rotary Position Embeddings use O(1)
  lookup tables instead of computing `powf`/`cosf`/`sinf` per token.
- **KV-Cache:** O(n) autoregressive inference with cached Key/Value
  states.

### Threading
- **Physical Core Pinning:** Automatic detection of physical vs
  logical cores, using only physical cores to avoid SMT contention.
- **Adaptive OpenMP:** Dynamic threshold (N≥512) activates
  parallelism only for large matrix operations (FFN, LM Head),
  avoiding thread synchronization overhead on small matrices.

## 📦 Project Structure
src/
├── tensor_int8.h # INT8 block quantization types
├── model.h # Model, RunState, KVCache definitions
├── matmul_blocked.h/cpp # AVX2 INT8 GEMV micro-kernel
├── layers.h/cpp # Transformer layers (Attention, FFN, RMSNorm)
├── quantize.cpp # AVX2-vectorized dynamic quantization
├── model.cpp # Weight loader (llama2.c format)
└── main.cpp # Inference loop with Prefill/Decode separation

text


## ⚙️ Build & Run

```bash
# Build
g++ -std=c++17 -O3 -march=native -mavx2 -mfma -fopenmp \
    src/quantize.cpp src/matmul_blocked.cpp src/layers.cpp \
    src/model.cpp src/main.cpp \
    -I./src -o mini_llm

# Run
./mini_llm models/stories15M.bin
📊 Architecture
text

Input Token
    │
    ▼
[Embedding Lookup]
    │
    ▼
┌──────────────────────── × N Layers ─┐
│  [RMSNorm] ──► [Multi-Head Attention]│
│       │              │               │
│       │    ┌─────────┘               │
│       │    │  Q·Kᵀ (AVX2 dot)       │
│       │    │  Softmax                │
│       │    │  Score×V (AVX2 FMA)     │
│       │    │  Output Proj (Wo)       │
│       │    └─────────┐               │
│       └── + Residual ◄               │
│              │                       │
│       [RMSNorm] ──► [FFN SwiGLU]    │
│              │         │             │
│              └── + Residual ◄        │
└──────────────────────────────────────┘
    │
    ▼
[RMSNorm] ──► [LM Head] ──► Argmax ──► Next Token
🔮 Roadmap
 FlashAttention-style Tiled Attention (CPU)
 CUDA port with Shared Memory Tiling
 Fused RMSNorm + RoPE CUDA kernel
 Tensor Core (WMMA) INT8 GEMM
 Tokenizer integration for interactive generation