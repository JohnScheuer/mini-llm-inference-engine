🚀 Mini‑LLM Inference Engine
High‑Performance Transformer Runtime (CPU + CUDA + Dynamic Serving)
A from‑scratch Transformer inference runtime written in modern C++17 and CUDA, designed to explore:

Low‑level Transformer execution
INT8 and FP16 kernel optimization
Tensor Core saturation
Dynamic micro‑batching
Queueing behavior under stochastic load
Roofline performance analysis
This project demonstrates hardware‑limited scaling and deterministic serving behavior on consumer‑grade GPUs.

⚡ Executive Performance Summary
Model	Precision	Peak Throughput	Sustained	P99 Latency	GPU
Stories110M	FP16	29.6k tok/s	32k tok/s	~134 ms	RTX 2070
TinyLlama 1.1B	INT8 (W8A8)	6.1k tok/s	~6k tok/s	Stable until knee	RTX 2070
✅ Hardware saturation achieved
✅ Near-roofline compute efficiency
✅ Stable tail latency under load
✅ Clear saturation knee observed

🖥 Hardware & Environment
GPU: NVIDIA RTX 2070 (Turing, SM 7.5)
CPU Backend: AVX2 INT8
CUDA: 12.x
Tensor Cores: FP16 + INT8
KV Cache: FP16
Serving Model: Poisson arrival process
Max Sequence Length: 1024
🧠 Architecture Overview
Core Design Principles
Fully custom Transformer runtime (no PyTorch execution)
Fused QKV projections
Fused FFN (W1 + W3)
True batched decode (no CPU loop per sequence)
Continuous dynamic batching
CUDA Graph multi‑bucket capture
Dual‑stream async IO
cuBLASLt optimized GEMM (FP16 + INT8)
📊 Stories110M (FP16 Tensor Core Backend)
🔹 Static Batch Scaling
Batch	Throughput
1	~600 tok/s
64	~11.8k tok/s
128	~17.8k tok/s
256	~24.4k tok/s
384	~29.6k tok/s (Peak)
Peak performance occurs near Batch ≈ 384, representing near‑full Tensor Core utilization.

🔹 Dynamic Micro‑Batching (Poisson Load)
Arrival Rate	Sustained Throughput
1000 req/s	~19k tok/s
2000 req/s	~28k tok/s
2300 req/s	~30k tok/s
2500 req/s	~32k tok/s
2600 req/s	~32k tok/s (Saturation begins)
📈 Tail Latency (P99)
<p align="center"> <img src="assets/benchmarks/latency_scaling_curve.png" width="800"/> </p>
Load	P50	P99	State
1000 req/s	76 ms	134 ms	Healthy
2300 req/s	76 ms	134 ms	Optimal
2500 req/s	80 ms	145 ms	SLA Limit
2600 req/s	112 ms	198 ms	Saturated
Interpretation
The system exhibits three clear regimes:

🟢 Underloaded – Compute underutilized
🟡 Optimal (Saturated) – GPU fully utilized, stable latency
🔴 Overloaded – Queue growth dominates latency

The sustainable serving capacity is approximately:

2400–2500 req/s

Beyond this point, the system transitions from compute‑bound to queue‑bound.

📊 Saturation Sweep (20k Request Marathon)
<p align="center"> <img src="assets/benchmarks/final_saturation_sweep.png" width="800"/> </p>
Peak Stable Throughput: 32,421 tok/s
Optimal Operating Point: ~2,500 req/s
Tail Latency: Stable <150ms across linear scaling region

This confirms the engine is hardware‑limited, not software‑limited.

🚀 TinyLlama 1.1B (Full INT8 W8A8)
After applying:

Full W8A8 quantization
cuBLASLt INT8 GEMM
Continuous batching
CUDA Graph capture
Fused attention pipeline
The engine reaches physical limits of the RTX 2070.

📊 Throughput vs Tail Latency
<p align="center"> <img src="assets/benchmarks/throughput_vs_latency.png" width="800"/> </p>
Key Observations
Linear scaling until ~60 req/s
Saturation knee at ~6.1k tok/s
Beyond knee → queueing delay increases exponentially
This is classic M/M/1 queue behavior under finite service capacity.

📈 Roofline Analysis
110M vs 1.1B Scaling
<p align="center"> <img src="assets/benchmarks/roofline_scaling.png" width="800"/> </p>
Measured Performance
Model	Precision	Performance
Stories110M	FP16	~420 GFLOPS
TinyLlama 1.1B	INT8	~782 GFLOPS
The INT8 model shifts right on the roofline (higher arithmetic intensity), operating deeper in the compute‑bound regime.

🏆 2.2 TFLOPS Breakthrough
<p align="center"> <img src="assets/benchmarks/roofline_breakthrough.png" width="800"/> </p>
2.2 TFLOPS sustained
~87% of achievable INT8 Tensor Core ceiling
Effective on‑chip bandwidth ≈ 1.1 TB/s (L2 reuse dominated)
This validates efficient Tensor Core utilization and memory reuse.

📊 SLA Stability & Saturation Behavior
<p align="center"> <img src="assets/benchmarks/sla_stability.png" width="800"/> </p>
Throughput scales to hardware limit (~26–32k tok/s depending on model)
P99 remains stable in compute‑bound regime
Clean transition into queue‑bound regime
No unstable oscillations or jitter amplification
The system maintains deterministic latency behavior under stochastic load.

🧠 System Regimes
The runtime clearly demonstrates three operational states:

🟢 Underloaded
Low arrival rate, minimal queue latency.

🟡 Compute‑Bound (Optimal)
GPU saturated, maximum throughput, stable SLA.

🔴 Queue‑Bound (Overload)
Arrival rate exceeds service capacity → exponential tail growth.

🏗 Architectural Highlights
✅ Custom Transformer runtime
✅ Fused QKV projection
✅ Fused FFN (W1 + W3)
✅ True batched decode
✅ Continuous adaptive scheduler
✅ Multi‑bucket CUDA Graph capture
✅ cuBLASLt INT8 optimization
✅ Roofline‑validated hardware saturation

🎯 Key Results
~130× speedup vs CPU backend
~29k tok/s peak (110M FP16)
~6k tok/s peak (1.1B INT8)
~32k tok/s sustained serving
Clear hardware saturation knee identified
Stable tail latency until overload
🏁 Conclusion
This project demonstrates:

Practical Tensor Core saturation
Real dynamic serving architecture
Queue‑aware system design
INT8 arithmetic intensity benefits
Roofline‑validated performance scaling
Deterministic SLA under load
The runtime approaches the physical throughput limits of an RTX 2070 for this workload.

Author: João Felipe De Souza


📜 License
MIT License