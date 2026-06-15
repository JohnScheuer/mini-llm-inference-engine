# Mini-LLM Inference Engine (HPC Focus)

Um motor de inferência Transformer "bare-metal" desenvolvido em C++17 puro, projetado para extrair o limite teórico de performance de CPUs x86_64 através de otimizações de baixo nível.

## 🚀 Performance
- **Throughput:** ~1,700 tokens/s (Modelo Stories15M)
- **Hardware:** AMD Ryzen 5 5600X (6-Cores, Zen 3)
- **Latência:** < 0.6ms por token

## 🛠️ Otimizações Implementadas
- **Quantização INT8 (W8A32):** Redução de 75% na largura de banda de memória e ocupação de cache.
- **Micro-kernel AVX2/FMA:** Uso intensivo de intrínsecas SIMD para processamento vetorial de 256 bits.
- **Cadeia de Instruções maddubs:** Implementação da instrução `vpmaddubsw` para quadruplicar o throughput de operações inteiras.
- **Manual 2x Unrolling:** Otimização de loop para explorar o *Instruction-Level Parallelism* (ILP) da arquitetura Zen 3.
- **L3 Cache Residency:** Otimizado para rodar inteiramente dentro do cache L3 (32MB), eliminando o gargalo da DRAM.
- **Paralelismo Inteligente:** Gerenciamento dinâmico de threads via OpenMP para mitigar overhead em matrizes pequenas.

## 📦 Estrutura do Projeto
- `src/matmul_blocked.cpp`: Core de computação SIMD.
- `src/quantize.cpp`: Motor de conversão dinâmica para INT8.
- `src/layers.cpp`: Implementação de camadas Transformer (Attention/FFN).
- `src/model.cpp`: Orquestrador do forward pass.

## ⚙️ Como Compilar
```bash
g++ -std=c++17 -O3 -march=native -mavx2 -mfma -fopenmp src/*.cpp -I./src -o mini_llm