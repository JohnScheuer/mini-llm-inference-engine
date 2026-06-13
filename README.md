# 🧠 Mini-LLM Inference Engine

[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)
[![CMake](https://img.shields.io/badge/CMake-3.10%2B-green.svg)](https://cmake.org/)
[![OpenMP](https://img.shields.io/badge/OpenMP-4.5-orange.svg)](https://www.openmp.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> ⚡ **Performance:** ~60 tokens/s CPU-only (Ryzen 5 5600X, AVX2 optimized, batch=1)

Motor de inferência **Transformer puro em C++**, escrito do zero, sem dependências de PyTorch, TensorFlow ou qualquer outro framework de ML.

Implementa toda a arquitetura moderna de LLMs (RoPE, RMSNorm, SwiGLU, KV-Cache) com foco em:

- ✅ Otimização SIMD (AVX2/FMA)
- ✅ Eficiência de cache
- ✅ Estrutura modular
- ✅ Controle preciso de paralelização

---

# ✨ Features

- 🎯 **Arquitetura Transformer completa**
  - Multi-Head Attention
  - RoPE
  - RMSNorm
  - FFN com SwiGLU
- 💾 **KV-Cache eficiente**
  - Complexidade O(n) por token
- 🗜️ **Quantização FP16**
- 🔤 **Tokenizer BPE próprio**
- ⚡ **SIMD AVX2 otimizado**
- 🎛️ **Sampling configurável**
- 📦 **Formato binário próprio**
- 🖥️ **CLI completa**

---

# 🏗️ Arquitetura
Embedding
↓
N × Transformer Block
↓
RMSNorm Final
↓
LM Head

text


Transformer Block:
x → RMSNorm → MHA → +residual
→ RMSNorm → FFN(SwiGLU) → +residual

text


---

# ⚡ CPU Optimizations (AVX2 / SIMD)

## ✅ AVX2 GEMM

Custom blocked matrix multiplication:

- `_mm256_fmadd_ps`
- Register blocking (6×16)
- Cache tiling
- FMA acceleration

Utilizado em:
- Projeções Q/K/V
- Output projection
- FFN layers

---

## ✅ Vectorized Attention

### 🔹 Q · K (Dot Product SIMD)

- AVX2 horizontal reduction
- 8 floats por iteração
- FMA acumulado

### 🔹 Weighted Sum com V (SIMD)

- Vetorizado com `_mm256_set1_ps`
- Acumulação AVX2
- Redução de tráfego de memória

---

## ✅ Parallelism Strategy

OpenMP é usado **apenas onde faz sentido**.

Nested parallelism foi removido porque:

- Batch=1 inference se beneficia mais de SIMD do que de oversubscription
- Overhead de threads reduzia performance
- Remoção do nested OpenMP quase dobrou throughput

Resultado final: ~60 tok/s CPU-only.

---

# 📊 Performance

### Hardware

- CPU: AMD Ryzen 5 5600X (6 cores)
- RAM: 32GB
- OS: Ubuntu (WSL)
- Compiler:
g++ -O3 -march=znver3 -mavx2 -mfma -fopenmp

text


### Model Configuration

| Parameter     | Value |
|--------------|-------|
| vocab_size   | 32000 |
| dim          | 512   |
| hidden_dim   | 2048  |
| n_layers     | 6     |
| n_heads      | 8     |
| head_dim     | 64    |
| max_seq_len  | 256   |

### Throughput (Batch=1 Autoregressive)

| Mode | Tokens/s |
|------|----------|
| Baseline OpenMP | ~22 tok/s |
| AVX2 + Attention SIMD | **~60 tok/s** |

Test:
- 128 tokens gerados
- KV-Cache habilitado
- Sem nested OpenMP

---

# 🧠 Why No Nested OpenMP?

Batch=1 inference é dominado por:

- Operações vetorizáveis
- Localidade de cache
- Overhead mínimo de threads

Nested OpenMP:

- Introduziu oversubscription
- Aumentou context switching
- Reduziu performance

Desabilitar nested parallelism quase dobrou throughput.

---

# 🚀 Build

```bash
git clone https://github.com/JohnScheuer/mini-llm-inference-engine.git
cd mini-llm-inference-engine
mkdir build && cd build
cmake ..
make -j$(nproc)
Ou compilação direta:

Bash

g++ -O3 -march=znver3 -mfma -mavx2 -fopenmp \
    -Isrc src/*.cpp \
    -o mini-llm-engine
▶️ Run
Bash

export OMP_NUM_THREADS=6
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_DYNAMIC=FALSE

./mini-llm-engine
📂 Project Structure
text

src/
 ├── tensor.h
 ├── layers.cpp
 ├── transformer.cpp
 ├── model.cpp
 ├── matmul_blocked.cpp
 ├── kernel_avx2.h
 ├── benchmark/
🎯 Roadmap
 KV-Cache
 RoPE
 SwiGLU
 FP16 Quantization
 AVX2 Optimization
 INT8 Quantization
 Flash-style Attention
 CUDA Backend
 AVX512 Support
📚 Referências
Attention Is All You Need (Vaswani et al., 2017)
RoFormer (Rotary Position Embedding)
GLU Variants Improve Transformer
RMSNorm
llama.cpp (inspiração)
👨‍💻 Autor
John Scheuer

<div align="center">
⭐ Se este projeto te ajudou, deixe uma estrela!

</div> ```