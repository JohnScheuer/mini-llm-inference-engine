#include "benchmark.h"
#include "../layers.h"
#include "../tensor.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <cstdlib>

// Inicializa um tensor com valores aleatórios determinísticos
static void init_random(Tensor& t, unsigned seed = 42) {
    std::srand(seed);
    for (auto& v : t.data) {
        v = (float)std::rand() / RAND_MAX - 0.5f;
    }
}

// Benchmark de MatMul para um tamanho NxN
BenchResult bench_matmul_naive(int N, int threads) {
    omp_set_num_threads(threads);
    
    Tensor A(N, N), B(N, N), C(N, N);
    init_random(A, 42);
    init_random(B, 43);
    
    // FLOPs = 2 * N^3 (cada elemento de C: N mul + (N-1) add ≈ 2N ops)
    long long flops = 2LL * N * N * N;
    
    // Bytes acessados = leitura de A + leitura de B + escrita de C
    long long bytes = (long long)N * N * sizeof(float) * 3;
    
    return Benchmark::run(
        "matmul_naive_" + std::to_string(N) + "x" + std::to_string(N) + "_t" + std::to_string(threads),
        [&]() { matmul_naive(C, A, B); },
        N,
        flops,
        bytes,
        {.warmup_runs = 2, .measured_runs = 5, .verbose = false}
    );
}

void run_matmul_benchmarks() {
    std::cout << "\n=================================================" << std::endl;
    std::cout << "  Benchmark: MatMul Naive (baseline)" << std::endl;
    std::cout << "=================================================" << std::endl;
    
    std::vector<int> sizes = {64, 128, 256, 512, 1024};
    std::vector<int> thread_counts = {1, 2, 4, 6, 8, 12};
    std::vector<BenchResult> results;
    
    // ===== TESTE 1: Diferentes tamanhos com max threads =====
    std::cout << "\n--- Teste 1: Escalando tamanho (12 threads) ---" << std::endl;
    for (int N : sizes) {
        std::cout << "  Rodando matmul " << N << "x" << N << "..." << std::flush;
        auto r = bench_matmul_naive(N, 12);
        results.push_back(r);
        std::cout << " " << std::fixed << std::setprecision(2) << r.gflops << " GFLOPS"
                  << " (" << (r.median_ns / 1e6) << " ms)" << std::endl;
    }
    
    // ===== TESTE 2: Scaling de threads em tamanho fixo =====
    std::cout << "\n--- Teste 2: Scaling de threads (matriz 512x512) ---" << std::endl;
    for (int t : thread_counts) {
        std::cout << "  Threads=" << t << "..." << std::flush;
        auto r = bench_matmul_naive(512, t);
        results.push_back(r);
        std::cout << " " << std::fixed << std::setprecision(2) << r.gflops << " GFLOPS"
                  << " (" << (r.median_ns / 1e6) << " ms)" << std::endl;
    }
    
    // ===== RESUMO =====
    Benchmark::print_table(results);
    
    // Calcula speedup
    std::cout << "\n--- Análise de Speedup (matriz 512x512) ---" << std::endl;
    double baseline_ms = 0;
    for (const auto& r : results) {
        if (r.problem_size == 512) {
            if (r.num_threads == 1) baseline_ms = r.median_ns / 1e6;
        }
    }
    
    std::cout << std::left
              << std::setw(10) << "Threads"
              << std::setw(15) << "Tempo (ms)"
              << std::setw(12) << "Speedup"
              << std::setw(15) << "Efficiency"
              << std::endl;
    std::cout << std::string(52, '-') << std::endl;
    
    for (const auto& r : results) {
        if (r.problem_size == 512) {
            double ms = r.median_ns / 1e6;
            double speedup = baseline_ms / ms;
            double efficiency = (speedup / r.num_threads) * 100.0;
            
            std::cout << std::left
                      << std::setw(10) << r.num_threads
                      << std::setw(15) << std::fixed << std::setprecision(3) << ms
                      << std::setw(12) << std::setprecision(2) << speedup << "x"
                      << std::setw(15) << std::setprecision(1) << efficiency << "%"
                      << std::endl;
        }
    }
    
    // Salva CSV
    Benchmark::save_csv(results, "bench_matmul_naive.csv");
}