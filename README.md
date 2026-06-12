# 🧠 Mini-LLM Inference Engine

[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)
[![CMake](https://img.shields.io/badge/CMake-3.10%2B-green.svg)](https://cmake.org/)
[![OpenMP](https://img.shields.io/badge/OpenMP-4.5-orange.svg)](https://www.openmp.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Motor de inferência **Transformer puro em C++**, escrito do zero, sem dependências de PyTorch, TensorFlow ou qualquer outro framework de ML. Implementa toda a arquitetura moderna de LLMs (RoPE, RMSNorm, SwiGLU, KV-Cache) com paralelização via OpenMP.

---

## ✨ Features

- 🎯 **Arquitetura Transformer completa** — Multi-Head Attention, RoPE, RMSNorm, FFN com SwiGLU
- 💾 **KV-Cache** — geração eficiente, complexidade O(n) ao invés de O(n²)
- 🗜️ **Quantização FP16** — modelos com metade do tamanho, sem perda perceptível
- 🔤 **Tokenizer BPE próprio** — encoder + decoder do zero, compatível com formato GPT-2/LLaMA
- ⚡ **Paralelização OpenMP** — aproveita todos os núcleos da CPU
- 🎛️ **Sampling configurável** — Argmax, Temperature, Top-K
- 📦 **Formato binário próprio** — header com magic number, versão e metadados
- 🖥️ **CLI completa** — argparse com help, validações e estatísticas

---

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────┐
│              CLI (argparse)                     │
├─────────────────────────────────────────────────┤
│         Tokenizer BPE (texto ↔ IDs)             │
├─────────────────────────────────────────────────┤
│           Sampling (Argmax / Top-K)             │
├─────────────────────────────────────────────────┤
│  Model: Embedding → N × Transformer Block       │
│         → RMSNorm Final → LM Head               │
├─────────────────────────────────────────────────┤
│  Transformer Block:                             │
│    x → RMSNorm → MHA → +residual                │
│      → RMSNorm → FFN(SwiGLU) → +residual        │
├─────────────────────────────────────────────────┤
│  Primitives: Tensor, MatMul (OpenMP), Softmax   │
│              RoPE, KV-Cache, FP16 Quantization  │
└─────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Pré-requisitos

- Compilador C++17 (GCC 9+, Clang 10+)
- CMake 3.10+
- OpenMP

No Ubuntu/WSL:
```bash
sudo apt update
sudo apt install build-essential cmake libomp-dev
```

### Build

```bash
git clone https://github.com/JohnScheuer/mini-llm-inference-engine.git
cd mini-llm-inference-engine
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Uso

```bash
./llm --model model.bin --prompt "hello world" --tokens 30
```

Exemplo completo com todas as opções:
```bash
./llm \
    --model model_fp16.bin \
    --prompt "hello world" \
    --tokens 50 \
    --temperature 0.8 \
    --top-k 10 \
    --vocab ../vocab/vocab.txt \
    --merges ../vocab/merges.txt
```

Ver ajuda completa:
```bash
./llm --help
```

---

## 📂 Estrutura do Projeto

```
mini-llm-interface/
├── CMakeLists.txt           # Build system
├── README.md
├── src/
│   ├── tensor.h             # Estrutura base (matriz 2D)
│   ├── layers.h/cpp         # RMSNorm, RoPE, MatMul, Softmax, Attention, FFN
│   ├── transformer.h/cpp    # Bloco Transformer com residuais
│   ├── model.h/cpp          # Model (embedding + blocks + LM head)
│   ├── generate.h/cpp       # Sampling + loop autoregressivo
│   ├── weights.h/cpp        # Save/Load de modelos .bin
│   ├── quantize.h/cpp       # FP32 ↔ FP16
│   ├── tokenizer.h/cpp      # BPE encoder/decoder
│   ├── cli.h/cpp            # Argparse
│   └── main.cpp             # Entry point
└── vocab/
    ├── vocab.txt            # Vocabulário (token por linha)
    └── merges.txt           # Regras de merge BPE
```

---

## 🧮 Detalhes Técnicos

### Multi-Head Attention com KV-Cache

Cada cabeça processa um sub-espaço `head_dim = dim / n_heads` do vetor. O KV-Cache armazena chaves e valores de todos os tokens passados, eliminando recomputação.

```
Q, K, V = MatMul(x, W_q/k/v)
Q, K = apply_rope(Q, K, pos)
cache[pos] = K, V
scores = Q · K^T / sqrt(head_dim)
attn = softmax(scores) · V
out = MatMul(attn, W_o)
```

**Complexidade:**
- Sem cache: O(n²) por token
- Com cache: O(n) por token

### Rotary Position Embedding (RoPE)

Codifica posição via rotação de pares de dimensões no plano complexo:

```
θ_i = pos / 10000^(2i/d)
[x_2i, x_2i+1] = [x_2i · cos(θ_i) - x_2i+1 · sin(θ_i),
                  x_2i · sin(θ_i) + x_2i+1 · cos(θ_i)]
```

Aplicado em Q e K antes do produto escalar.

### SwiGLU FFN

```
out = W_2 · (SiLU(W_1 · x) ⊙ (W_3 · x))
```

Onde `SiLU(x) = x · σ(x)` e `⊙` é multiplicação elemento-a-elemento.

### Quantização FP16

Conversão manual bit-a-bit entre IEEE 754 FP32 e FP16:

| Tipo | Bits | Range | Precisão |
|------|------|-------|----------|
| FP32 | 32 | ±3.4e38 | 7 dígitos |
| FP16 | 16 | ±65504  | 3-4 dígitos |

Resultado: **modelos com 50% do tamanho** sem perda perceptível em inferência.

### Tokenizer BPE

Implementação completa do algoritmo Byte Pair Encoding:

1. **Pré-tokenização** por espaços
2. **Divisão** em caracteres
3. **Merges iterativos** seguindo ordem de prioridade do `merges.txt`
4. **Conversão** para IDs via `vocab.txt`

Compatível com o formato usado por GPT-2 e LLaMA.

---

## 📊 Performance

Testado em **Intel i7 (12 threads), modelo dim=16, layers=2**:

| Operação | Tempo |
|----------|-------|
| Carregamento do modelo | ~1 ms |
| Forward pass por token | ~0.3 ms |
| Throughput de geração | ~3000 tokens/s |

Em modelos maiores, o ganho do OpenMP escala com o tamanho do `dim` e o número de cabeças.

---

## 🎯 Roadmap

- [x] Tensor base + MatMul paralelizado (OpenMP)
- [x] RMSNorm
- [x] RoPE
- [x] Multi-Head Attention + KV-Cache
- [x] FFN com SwiGLU
- [x] Transformer Block com conexões residuais
- [x] Modelo completo (Embedding + N Blocks + LM Head)
- [x] Sampling (Argmax, Temperature, Top-K)
- [x] Save/Load formato binário próprio
- [x] Quantização FP16
- [x] Tokenizer BPE
- [x] CLI profissional
- [ ] Quantização INT8 com escala por tensor
- [ ] Suporte a Grouped-Query Attention (GQA)
- [ ] Carregamento de pesos do GPT-2/TinyLlama via script de conversão
- [ ] Otimização SIMD (AVX2/AVX512)
- [ ] Suporte a modelos com tied embeddings

---

## 📚 Referências

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [GLU Variants Improve Transformer (Shazeer, 2020)](https://arxiv.org/abs/2002.05202)
- [Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)](https://arxiv.org/abs/1910.07467)
- [Neural Machine Translation of Rare Words with Subword Units (BPE, Sennrich et al., 2016)](https://arxiv.org/abs/1508.07909)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — inspiração principal

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se livre para abrir issues ou pull requests.

Áreas que precisam de ajuda:
- Implementação de INT8 quantization
- Otimizações SIMD
- Testes unitários
- Documentação

---

## 📝 Licença

Este projeto está licenciado sob a [MIT License](LICENSE) — sinta-se livre para usar, modificar e distribuir.

---

## 👨‍💻 Autor

**John Scheuer**

- GitHub: [@JohnScheuer](https://github.com/JohnScheuer)

---

<div align="center">

**⭐ Se este projeto te ajudou, deixe uma estrela no repositório!**

Feito com ❤️ e muito C++ puro

</div>