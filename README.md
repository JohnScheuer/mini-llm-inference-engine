# 🧠 Mini-LLM Inference Engine (C++ / AVX2)

> ⚡ ~420 tokens/s CPU-only (Ryzen 5 5600X)  
> 🚀 Transformer completo em C++ com AVX2, KV-Cache e modelo real Stories15M

Motor de inferência Transformer implementado **do zero em C++**, sem PyTorch, TensorFlow ou qualquer framework de ML.

Compatível com o modelo **Stories15M (llama2.c)** e tokenizer real (`tokenizer.bin`).

---

## 🎯 Objetivo do Projeto

Construir um engine de inferência de LLM:

- ✅ Alto desempenho em CPU
- ✅ Arquitetura limpa e modular
- ✅ Controle total de memória
- ✅ Otimização SIMD (AVX2/FMA)
- ✅ Compatível com modelo real

---

# 🚀 Performance

### Hardware
- CPU: AMD Ryzen 5 5600X (6 cores)
- RAM: 32GB
- OS: Ubuntu (WSL)
- Compilação:
g++ -O3 -march=znver3 -mavx2 -mfma -fopenmp

text


### Modelo
- Stories15M (llama2.c)
- dim = 288
- hidden_dim = 768
- layers = 6
- heads = 6
- vocab_size = 32000

### Throughput

| Configuração | Tokens/s |
|--------------|----------|
| Baseline inicial | ~22 tok/s |
| AVX2 matmul + attention SIMD | ~60 tok/s |
| Versão final (sem nested OpenMP) | **~420 tok/s** |

Teste:
- 100 tokens autoregressivos
- KV-cache ativo
- Batch = 1

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


### Transformer Block
x → RMSNorm → Multi-Head Attention → +residual
→ RMSNorm → SwiGLU FFN → +residual

text


---

# ⚡ Otimizações Implementadas

## ✅ AVX2 GEMM (6×16 Register Blocking)
- `_mm256_fmadd_ps`
- Cache tiling
- Layout cache-friendly
- FMA acceleration

## ✅ Attention SIMD
- AVX2 dot product (Q·K)
- AVX2 weighted sum com V
- Redução horizontal vetorizada

## ✅ KV-Cache
- Complexidade O(n) por token
- Armazenamento eficiente
- Sem recomputação de chaves/valores

## ✅ Loader Binário Real
- Leitura correta do header (llama2.c)
- Ordem intercalada de camadas
- Suporte a embeddings compartilhados

## ✅ Paralelização Controlada
- Nested OpenMP removido
- Melhor uso de SIMD ao invés de oversubscription
- Redução significativa de overhead

---

# 📂 Estrutura do Projeto
src/
├── tensor.h
├── layers.cpp
├── transformer.cpp
├── model.cpp
├── matmul_blocked.cpp
├── kernel_avx2.h
├── tokenizer.cpp
└── main.cpp

text


---

# 🔧 Build

```bash
git clone <repo>
cd mini-llm-interface

g++ -O3 -march=znver3 -mfma -mavx2 -fopenmp \
    -Isrc src/*.cpp \
    -o mini-llm-engine
▶️ Run
Bash

export OMP_NUM_THREADS=6
export OMP_PROC_BIND=true
export OMP_PLACES=cores

./mini-llm-engine
📌 Exemplo de Saída
text

Once upon a time, there was a little girl named Lily...
🧠 Engenharia de Performance
Principais decisões técnicas:

SIMD > paralelização ingênua (batch=1)
Remoção de nested OpenMP
Alinhamento correto de layout binário
Controle manual de leitura de pesos
Diagnóstico com AddressSanitizer
Redução de alocações dinâmicas
📊 Evolução do Projeto
Implementação ingênua → ~22 tok/s
Matmul bloqueado AVX2 → ~60 tok/s
Attention vetorizado → ~100+ tok/s
Remoção de nested OpenMP → ~400+ tok/s
Integração modelo real Stories15M
🛣️ Roadmap
 Transformer completo
 KV-cache
 AVX2 SIMD
 Loader binário compatível
 INT8 quantization
 Flash-style attention
 AVX512
 Backend CUDA
📚 Referências
Attention Is All You Need
RoPE
SwiGLU
llama2.c (Karpathy)
TinyStories
👨‍💻 Autor
John Scheuer

⭐ Se este projeto te ajudou, deixe uma estrela!
