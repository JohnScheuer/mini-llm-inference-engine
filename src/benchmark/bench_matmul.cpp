#include "benchmark.h"
#include "../layers.h"
#include "../tensor.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <cstdlib>
#include <cmath>

static void init_random(Tensor& t, unsigned seed = 42) {
    std::srand(seed);
    for (auto& v : t.data) {
        v = (float)std::rand() / RAND_MAX - 0.5f;
    }
}

static bool verify(const Tensor& C1, const Tensor& C2, float tol = 1e-3f) {
    if (C1.rows != C2.rows || C1.cols != C2.cols) return false;
    for (size_t i = 0; i < C1.data.size(); i++) {
        if (std::abs(C1.data[i] - C2.data[i]) > tol) {
            return false;
        }
    }
    return true;
}

void run_matmul_benchmarks() {
    std::cout << "\n=================================================" << std::endl;
    std::cout << "  Benchmark: MatMul (Naive vs Otimizadas)" << std::endl;
    std::cout << "=================================================" << std::endl;
    
    omp_set_num_threads(12);
    
    std::vector<int> sizes = {128, 256, 512, 1024};
    std::vector<BenchResult> results;
    
    for (int N : sizes) {
        std::cout << "\n--- Matriz " << N << "x" << N << " ---" << std::endl;
        
        Tensor A(N, N), B(N, N);
        Tensor C_naive(N, N), C_ikj(N, N), C_tiled(N, N), C_tiled_ikj(N, N);
        init_random(A, 42);
        init_random(B, 43);
        
        long long flops = 2LL * N * N * N;
        long long bytes = (long long)N * N * sizeof(float) * 3;
        BenchConfig cfg;
        cfg.warmup_runs = 1;
        cfg.measured_runs = 3;
        cfg.verbose = false;
        
        // Naive
        auto r_naive = Benchmark::run(
            "matmul_naive_" + std::to_string(N),
            [&]() { matmul_naive(C_naive, A, B); },
            N, flops, bytes, cfg
        );
        results.push_back(r_naive);
        std::cout << "  naive:        " << std::setw(7) << std::fixed << std::setprecision(2)
                  << r_naive.gflops << " GFLOPS  (" 
                  << std::setprecision(1) << (r_naive.median_ns / 1e6) << " ms)" << std::endl;
        
        // i-k-j
        auto r_ikj = Benchmark::run(
            "matmul_ikj_" + std::to_string(N),
            [&]() { matmul_ikj(C_ikj, A, B); },
            N, flops, bytes, cfg
        );
        results.push_back(r_ikj);
        double speedup_ikj = r_naive.median_ns / r_ikj.median_ns;
        std::cout << "  ikj:          " << std::setw(7) << std::setprecision(2)
                  << r_ikj.gflops << " GFLOPS  ("
                  << std::setprecision(1) << (r_ikj.median_ns / 1e6) << " ms)"
                  << "  [" << std::setprecision(2) << speedup_ikj << "x vs naive]" << std::endl;
        
        // Tiled
        auto r_tiled = Benchmark::run(
            "matmul_tiled64_" + std::to_string(N),
            [&]() { matmul_tiled(C_tiled, A, B, 64); },
            N, flops, bytes, cfg
        );
        results.push_back(r_tiled);
        double speedup_tiled = r_naive.median_ns / r_tiled.median_ns;
        std::cout << "  tiled(T=64):  " << std::setw(7) << std::setprecision(2)
                  << r_tiled.gflops << " GFLOPS  ("
                  << std::setprecision(1) << (r_tiled.median_ns / 1e6) << " ms)"
                  << "  [" << std::setprecision(2) << speedup_tiled << "x vs naive]" << std::endl;
        
        // Tiled + i-k-j
        auto r_combo = Benchmark::run(
            "matmul_tiled_ikj_" + std::to_string(N),
            [&]() { matmul_tiled_ikj(C_tiled_ikj, A, B, 64); },
            N, flops, bytes, cfg
        );
        results.push_back(r_combo);
        double speedup_combo = r_naive.median_ns / r_combo.median_ns;
        std::cout << "  tiled+ikj:    " << std::setw(7) << std::setprecision(2)
                  << r_combo.gflops << " GFLOPS  ("
                  << std::setprecision(1) << (r_combo.median_ns / 1e6) << " ms)"
                  << "  [" << std::setprecision(2) << speedup_combo << "x vs naive] *" << std::endl;
        
        bool ok_ikj = verify(C_naive, C_ikj);
        bool ok_tiled = verify(C_naive, C_tiled);
        bool ok_combo = verify(C_naive, C_tiled_ikj);
        std::cout << "  Correcao:     ikj=" << (ok_ikj ? "OK" : "FAIL")
                  << "  tiled=" << (ok_tiled ? "OK" : "FAIL")
                  << "  combo=" << (ok_combo ? "OK" : "FAIL") << std::endl;
    }
    
    // Análise: melhor tile size
    std::cout << "\n--- Analise: tile size otimo (matriz 512x512) ---" << std::endl;
    Tensor A(512, 512), B(512, 512), C(512, 512);
    init_random(A, 42);
    init_random(B, 43);
    
    BenchConfig cfg2;
    cfg2.warmup_runs = 1;
    cfg2.measured_runs = 3;
    cfg2.verbose = false;
    
    for (int T : {16, 32, 64, 128, 256}) {
        auto r = Benchmark::run(
            "matmul_tiled_ikj_T" + std::to_string(T),
            [&]() { matmul_tiled_ikj(C, A, B, T); },
            512, 2LL * 512 * 512 * 512, 0,
            cfg2
        );
        results.push_back(r);
        std::cout << "  T=" << std::setw(4) << T << ":  " 
                  << std::setw(7) << std::fixed << std::setprecision(2) << r.gflops 
                  << " GFLOPS  (" << std::setprecision(1) << (r.median_ns / 1e6) << " ms)" 
                  << std::endl;
    }
    
    Benchmark::save_csv(results, "bench_matmul_comparison.csv");
}