# 🚀 Mini‑LLM Inference Engine
### High‑Performance Transformer Runtime (CPU + GPU + Dynamic Serving)

A from‑scratch Transformer inference runtime written in modern C++ and CUDA, optimized for:

- ✅ AVX2 INT8 CPU execution
- ✅ FP16 Tensor Core GPU acceleration
- ✅ Fused QKV and FFN projections
- ✅ True batched decode inference
- ✅ Adaptive dynamic micro‑batching
- ✅ CUDA Graph launch optimization

This project demonstrates real hardware saturation and serving‑level inference performance.

---

# 📌 System Configuration

**GPU:** NVIDIA RTX 2070 (SM 7.5)  
**CUDA:** 13.2  
**Precision:** FP16 (Tensor Cores)  
**Model Tested:** Stories110M (110M parameters)  
**Max Sequence Length:** 1024  

---

# ⚡ Performance Results

## 🔹 CPU Backend (INT8 AVX2)

| Model | Batch | Throughput |
|--------|--------|-------------|
| Stories110M | 1 | ~228 tok/s |

---

## 🔹 GPU Backend (FP16 Tensor Cores)

### Batch Scaling – Stories110M

| Batch Size | Throughput (tok/s) |
|------------|--------------------|
| 1          | ~600 |
| 16         | ~3,700 |
| 64         | ~11,851 |
| 96         | ~16,141 |
| 128        | ~17,846 |
| 160        | ~19,799 |
| 192        | ~22,859 |
| 256        | ~24,483 |
| 320        | ~27,833 |
| 384        | **~29,650 (Peak)** |
| 416        | ~28,852 |
| 448        | ~27,553 |

Peak throughput occurs near **Batch ≈ 384**, representing full Tensor Core saturation.

---

# 🔁 Adaptive Dynamic Micro‑Batching

Simulated serving workload with 1000 concurrent requests:

| Mode | Throughput |
|------|------------|
| Static Batch=192 | ~22,859 tok/s |
| Adaptive Dynamic | **~24,482 tok/s** |

Adaptive scheduler maintains ~82–85% of peak static throughput while supporting dynamic request completion.

---

# 🧠 Architectural Highlights

## ✅ Fused QKV Projection
Single GEMM computing Q, K, V simultaneously.

## ✅ Fused FFN (W1 + W3)
Reduced kernel launches and improved Tensor Core utilization.

## ✅ True Batched Decode
- GEMM dimension N = Batch
- No CPU loop per sequence
- Full GPU parallelism

## ✅ CUDA Graph Integration
Reduced launch overhead in dynamic serving mode.

## ✅ Continuous Scheduler
- Slot reuse
- Micro-batch adaptation
- Dynamic workload balancing

---

# 📈 Scaling Behavior

- Throughput scales near-linearly until GPU saturation.
- Peak performance observed at Batch ≈ 384.
- Beyond saturation, memory bandwidth and scheduler contention reduce efficiency.

---

# 🎯 CPU vs GPU Comparison

| Backend | Batch | Throughput |
|----------|--------|-------------|
| CPU INT8 | 1 | ~228 tok/s |
| GPU FP16 | 1 | ~600 tok/s |
| GPU FP16 | 384 | ~29,650 tok/s |
| GPU FP16 Dynamic | ~192 | ~24,482 tok/s |

GPU provides **~130× throughput improvement over CPU** at optimal batch.

---

# 🏁 Conclusion

This project demonstrates:

- Practical GPU saturation analysis
- Tensor Core utilization strategies
- Batch scaling characterization
- Dynamic serving architecture
- Hardware limit discovery on RTX 2070

The runtime approaches the physical throughput limit of the GPU for this workload.

---

# 📜 License

MIT License