🧠 Mini-LLM Inference Engine
C++
Performance
GFLOPS
License

Um motor de inferência para Large Language Models (LLMs) desenvolvido do zero em C++, focado em performance extrema e otimização de baixo nível para microarquiteturas x86_64 (Zen 3).

O projeto implementa a arquitetura moderna de Transformers (Llama-style) utilizando kernels manuais em SIMD, superando drasticamente as otimizações padrão de compiladores e atingindo eficiência de nível industrial.

📊 Performance Benchmarks
Resultados obtidos em um AMD Ryzen 5 5600X (6 Cores / 12 Threads) @ 4.6GHz.

Métrica	Resultado	Cenário
Inference Throughput	421.3 tokens/s	Stories15M (Dim=288, Layers=6)
Peak Compute	321.8 GFLOPS	FP32 MatMul (1024x1024)
SIMD Efficiency	~70% do Pico Teórico	Manual AVX2/FMA Kernels
Latency (M=1)	~2.3 ms/token	Single-token generation
⚡ Otimizações HPC (High Performance Computing)
O motor não utiliza bibliotecas externas (como OpenBLAS ou MKL), baseando-se em implementações manuais de alto desempenho:

1. Matrix Multiplication (GEMM) & Vector-Matrix (GEMV)
Register Blocking (6x16): Micro-kernel desenvolvido em intrínsecos AVX2 que mantém um bloco de 96 elementos da matriz C nos registradores YMM, minimizando o tráfego de memória.
Hierarchical Tiling: Estratégia de blocagem de cache para níveis L1, L2 e L3.
Matrix-Vector Shortcut: Caminho de execução específico para inferência (
M
=
1
M=1), transformando a operação em um varrimento de memória linear ultra-otimizado.
2. Attention SIMD Pipeline
Vetorização de Scores: Cálculo de similaridade 
Q
⋅
K
Q⋅K utilizando produtos escalares paralelos em AVX2.
SIMD Weighted Sum: Agregação de valores 
V
V via instruções _mm256_fmadd_ps, otimizando o gargalo de memória do KV-Cache.
3. Gerenciamento de Threads e Memória
Thread Affinity: Fixação de threads em núcleos físicos via OMP_PROC_BIND=true para evitar migração de contexto e poluição de cache.
Zero-Allocation Loop: Pipeline de geração "malloc-free", eliminando o overhead de gerenciamento de memória do sistema operacional durante a inferência.
🏗️ Arquitetura do Modelo
RMSNorm: Normalização de alta estabilidade numérica.
RoPE (Rotary Positional Embeddings): Codificação de posição moderna para suporte a contextos longos.
SwiGLU Activation: Implementação eficiente da função de ativação Gated Linear Unit.
KV-Cache: Implementação O(n) para inferência autoregressiva.
🚀 Como Executar
Compilação de Alta Performance
Bash

# Compilar o engine otimizado para Zen 3 (Ryzen)
g++ -O3 -march=znver3 -mfma -mavx2 -fopenmp -Isrc src/*.cpp -o mini-llm-engine
Execução com Afinidade de Núcleos
Para atingir os 400+ tokens/s, é crucial prender as threads nos núcleos físicos:

C++

# Configuração para Ryzen 5 5600X (6 núcleos físicos)
export OMP_NUM_THREADS=6
export OMP_PROC_BIND=true
export OMP_PLACES=cores

./mini-llm-engine
📂 Estrutura do Projeto
src/kernel_avx2.h: Micro-kernels de baixo nível e intrínsecos SIMD.
src/matmul_blocked.cpp: Implementação GEMM/GEMV de alto desempenho.
src/layers.cpp: Camadas do Transformer (Attention, RMSNorm, FFN) vetorizadas.
src/tokenizer.cpp: Leitor binário de vocabulário (Padrão Llama 2).
src/model.cpp: Carregador de pesos e lógica de Forward Pass.
🗺️ Roadmap
 Kernels AVX2/FMA (321 GFLOPS)
 Inferência em Tempo Real (421 tokens/s)
 Suporte ao formato Llama 2 (.bin)
 Quantização INT8/Q4_0 (Meta: 1000+ tokens/s)
 Flash-style Attention (Otimização de banda de memória)
 Interface de Chat Interativa
📄 Licença
Distribuído sob a licença MIT. Veja LICENSE para mais informações.

Autor: John Scheuer
Hardware: Ryzen 5 5600X | 32GB RAM | WSL2 Ubuntu


