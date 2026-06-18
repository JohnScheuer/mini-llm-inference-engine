🧠 Mini-LLM Inference Engine: Bare-Metal Performance
C++
CUDA
Architecture

A standalone, high-performance LLM inference engine written from scratch in pure C++17 and CUDA. This project bypasses high-level frameworks (PyTorch/TensorFlow) to extract the absolute physical limits of consumer hardware through manual kernel fusion, SIMD/WMMA intrinsics, and hardware-aware memory orchestration.

🚀 Key Performance Milestones
Peak Throughput: 29,649.6 tokens/s on a 110M parameter model (NVIDIA RTX 2070).
Ultra-Low Latency: 0.35ms per token (CPU/INT8) and 0.57ms per token (GPU/FP16).
Massive Compute Density: Sustained 3.26 TFLOPS on Turing Tensor Cores.
Memory Mastery: Successfully orchestrated 14.4GB workloads on an 8GB hardware limit (Unified Memory Overcommit) with zero performance degradation.
📊 Performance Analysis
1. Throughput Scaling: CPU vs. GPU
The crossover point between CPU (Compute-Bound) and GPU (Memory-Bound) is mapped below using the Stories110M model.

Hardware Backend	Batch Size	Throughput	Latency	Optimization
CPU (Ryzen 5600X)	1	228.6 tok/s	4.38 ms	AVX2 / FMA / L3-Bound
GPU (RTX 2070)	1	596.9 tok/s	1.68 ms	CUDA FP16
GPU (RTX 2070)	384	29,649.6 tok/s	0.57 ms	WMMA Tensor Cores
2. Model Scaling (Stories Series)
Benchmarked on RTX 2070 using FP16 Precision.

mermaid

barChart
    title Throughput per Model Size (tok/s)
    x-axis Model Size (Parameters)
    y-axis tok/s
    "Stories15M": 2411
    "Stories42M": 645
    "Stories110M": 596
(Note: At 15M parameters, the model is entirely L3-cache resident, resulting in peak latency efficiency.)

3. The Batch Saturaion Curve
Demonstrating the transition from Memory-Bound to Compute-Bound execution.

Batch Size	Throughput (tok/s)	Memory Footprint	Efficiency Gain
1	596.9	220 MB	Baseline
16	3,707.9	1.1 GB	6.2x
64	11,851.3	3.2 GB	19.8x
128	17,845.9	5.5 GB	29.8x
384	29,649.6	14.4 GB	49.6x
🛠️ Technical Stack & Optimizations
GPU Backend (CUDA/WMMA)
Warp Matrix Multiply-Accumulate (WMMA): Manual utilization of Tensor Cores for 16x16x16 mixed-precision fragments.
FlashAttention-2D: Custom IO-aware attention kernel with Online Softmax to maintain 
O
(
1
)
O(1) memory overhead.
Kernel Fusion: Super-fused operators (Residual + RMSNorm) to eliminate memory round-trips between the SMs and VRAM.
Weight Packing: Concatention of WQ, WK, and WV weights into a single QKV GEMM launch to minimize PCIe/Driver overhead.
CPU Backend (x86_64 SIMD)
INT8 Quantization (W8A32): Manual vpmaddubsw chain implementation for 8-bit weights with 32-bit accumulation.
Hyper-Threading Optimization: Identified that disabling nested OpenMP and pinning threads to physical cores (6 threads vs 12) results in a 10x stability gain for memory-intensive workloads.
Zero-Allocation Hot Path: Pre-allocated RunState buffers to ensure zero malloc calls during the autoregressive loop.
System Observability
OS-Level Determinism: Validated via perf a state of 0 context-switches and 0 CPU migrations during peak load.
Unified Memory Orchestration: Engineered memory-access patterns to mask PCIe bandwidth bottlenecks during 180% VRAM overcommit scenarios.
🏗️ Architecture
mermaid

graph TD
    A[Input Tokens] --> B[Embedding Kernel]
    B --> C[Fused Residual + RMSNorm]
    C --> D[Fused QKV GEMM - cuBLAS/WMMA]
    D --> E[FlashAttention-2D + Online Softmax]
    E --> F[Output Projection]
    F --> G[Fused W13 GEMM + SiLU Activation]
    G --> H[W2 Projection]
    H --> I[Final RMSNorm]
    I --> J[GPU-Side Argmax]
    J --> K[Next Token]
🚀 Getting Started
Compilation
Bash

# GPU Backend
nvcc -std=c++17 -O3 -arch=sm_75 -ccbin /usr/bin/g++-12 \
    -I./src src/cuda_llm_batched.cu src/model.cpp \
    -lcublas -o mini_llm_gpu

# CPU Backend
g++ -std=c++17 -O3 -march=native -mfma -mavx2 -fopenmp \
    -I./src src/*.cpp -o mini_llm_cpu
Execution (Peak Throughput Mode)
Bash

./mini_llm_gpu models/stories110M.bin
👨‍💻 Author
John Scheuer
Senior Software Engineer | HPC & LLM Infrastructure
[\[LinkedIn Profile\]](https://www.linkedin.com/in/joaofelipescheuer/) | 

<div align="center"> ⭐ If you found this project useful for HPC research, please leave a star! </div>