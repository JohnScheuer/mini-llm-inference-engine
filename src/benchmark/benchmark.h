#pragma once
#include <chrono>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <iostream>
#include <fstream>

// Resultado de UM benchmark
struct BenchResult {
    std::string name;
    int problem_size;       // tamanho do problema (ex: dim da matriz)
    int num_threads;
    double median_ns;       // tempo mediano em nanosegundos
    double min_ns;          // melhor tempo (lower bound)
    double max_ns;          // pior tempo
    double std_dev_ns;      // desvio padrão
    double gflops;          // performance em GFLOPS
    double bandwidth_gbs;   // largura de banda em GB/s
    int warmup_runs;
    int measured_runs;
};

// Configuração padrão de um benchmark
struct BenchConfig {
    int warmup_runs = 3;
    int measured_runs = 10;
    bool verbose = true;
};

// Classe principal: roda um benchmark e retorna estatísticas
class Benchmark {
public:
    // Roda uma função N vezes, mede tudo
    // - flop_count: número de operações FP que a função faz (pra GFLOPS)
    // - bytes_accessed: bytes lidos/escritos (pra bandwidth)
    static BenchResult run(
        const std::string& name,
        std::function<void()> func,
        int problem_size,
        long long flop_count = 0,
        long long bytes_accessed = 0,
        BenchConfig config = {}
    );

    // Imprime resultado bonito no terminal
    static void print(const BenchResult& result);
    
    // Imprime tabela de comparação entre múltiplos resultados
    static void print_table(const std::vector<BenchResult>& results);

    // Salva resultados em CSV
    static void save_csv(
        const std::vector<BenchResult>& results,
        const std::string& filepath
    );
};