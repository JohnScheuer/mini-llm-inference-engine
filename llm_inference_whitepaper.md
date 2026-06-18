---
title: "Engineering High-Throughput Transformer Inference"
author: "João Felipe de Souza"
date: "June 2026"
geometry: margin=1in
fontsize: 11pt
---

# Abstract

This document presents the design, optimization, and performance characterization of a custom Transformer inference runtime built in C++ and CUDA. The system evolved from a CPU-bound baseline (~600 tokens/sec) into a GPU-saturated serving engine reaching ~30,000 tokens/sec on an RTX 2070 using FP16 Tensor Cores. Beyond peak performance, the runtime models realistic serving behavior under stochastic load and analyzes tail latency under saturation.

---

# 1. Introduction

Transformer inference performance depends heavily on hardware utilization. While many implementations focus on kernel-level micro-optimizations, real throughput improvements often require architectural changes such as batching and scheduling.

This work explores the full performance envelope of a 110M-parameter Transformer model on consumer-grade hardware.

---

# 2. System Overview

The runtime implements:

- Autoregressive decode
- KV cache
- Fused QKV projection
- Fused FFN (W1 + W3)
- FP16 Tensor Core acceleration
- Dynamic micro-batching scheduler
- Paged KV memory management
- Tail latency measurement (P50 / P95 / P99)

Hardware:
- NVIDIA RTX 2070 (SM 7.5)
- CUDA 13.x
- FP16 Tensor Cores

---

# 3. Static Batch Scaling

Throughput scaling (Stories110M):

| Batch | Throughput (tok/s) |
|-------|--------------------|
| 1     | ~600 |
| 16    | ~3,700 |
| 64    | ~11,851 |
| 128   | ~17,846 |
| 192   | ~22,859 |
| 256   | ~24,483 |
| 320   | ~27,833 |
| 384   | ~29,650 |

Peak performance occurs near Batch ≈ 384.

---

# 4. Dynamic Serving Simulation

Poisson arrival model was used to simulate real serving conditions.

Sustainable serving capacity:

| Req/s | Throughput | P99 Latency |
|-------|------------|-------------|
| 2300  | ~30k tok/s | ~134 ms |
| 2500  | ~32k tok/s | ~145 ms |
| 2600  | ~32k tok/s | ~198 ms |

Capacity threshold observed near 2400–2500 req/s.

---

# 5. Paged KV Cache

To support dynamic batching and variable-length requests, KV memory was reorganized into fixed-size pages.

Benefits:
- Reduced memory movement
- Improved flexibility
- Better serving realism

Throughput impact:

| Mode | Throughput |
|------|------------|
| Static batch | ~29k tok/s |
| Adaptive batching | ~24k tok/s |
| Paged-HPC Ultra | ~25k tok/s |

Paged architecture recovered most of the throughput while enabling realistic serving behavior.

---

# 6. Tail Latency Analysis

At 2300 req/s:

- P50 ≈ 76 ms
- P95 ≈ 129 ms
- P99 ≈ 134 ms

At 2600 req/s:

- P99 increases significantly (~198 ms)
- System enters overload regime

Behavior matches classical queueing theory under saturation.

---

# 7. Conclusions

Key findings:

- Batch size dominates GPU utilization.
- cuBLAS Tensor Core GEMM is extremely difficult to outperform.
- Dynamic micro-batching retains ~80% of peak throughput.
- RTX 2070 physical throughput ceiling ≈ 30–32k tok/s.
- Sustainable serving capacity ≈ 2400–2500 req/s.
- Tail latency grows predictably beyond capacity threshold.

This project demonstrates end-to-end performance engineering, GPU saturation analysis, and serving capacity modeling on commodity hardware.

---

# 8. Future Work

- KV cache compaction
- Multi-stream overlap
- Prefill vs decode separation
- Multi-GPU scaling
- Lower precision (FP8 / INT8 Tensor Core)

---

# References

NVIDIA CUDA Programming Guide  
Transformer Architecture (Vaswani et al.)  
Queueing Theory Fundamentals  
