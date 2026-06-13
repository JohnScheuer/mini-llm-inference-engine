#include "benchmark.h"
#include "../layers.h"
#include "../tensor.h"
#include "../matmul_blocked.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <cstdlib>
#include <cmath>
#include <cstring>

static void init_random(Tensor& t, unsigned seed = 42) {
    std::srand(seed);
    for (auto& v : t.data) {
        v = (float)std::rand() / RAND_MAX - 0.5f;
    }
}

static bool verify(const Tensor& C1, const Tensor& C2, float tol = 1e-2f) {
    if (C1.rows != C2.rows || C1.cols != C2.cols) return false;
    for (size_t i = 0; i < C1.data.size(); i++) {
        if (std::abs(C1.data[i] - C2.data[i]) > tol) {
            return false;
        }
    }
    return true;
}

static void print_result(const std::string& name, double gflops, double ms, 
                          double baseline_ms, bool correct) {
    double speedup = baseline_ms / ms;
    std::cout << "  " << std::left << std::setw(20) << name 
              << std::right << std::setw(7) << std::fixed << std::setprecision(2)
              << gflops << " GFLOPS  ("
              << std::setw(6) << std::setprecision(1) << ms << " ms)"
              << "  [" << std::setprecision(2) << speedup << "x]"
              << "  " << (correct ? "OK" : "FAIL")
              << std::endl;
}

void run_matmul_benchmarks() {
    std::cout << "\n=================================================" << std::endl;
    std::cout << "  Benchmark: MatMul (todas as versoes)" << std::endl;
    std::cout << "=================================================" << std::endl;
    
    omp_set_num_threads(12);
    
    std::vector<int> sizes = {256, 512, 1024};
    std::vector<BenchResult> results;
    
    for (int N : sizes) {
        std::cout << "\n--- Matriz " << N << "x" << N << " ---" << std::endl;
        
        Tensor A(N, N), B(N, N);
        Tensor C_naive(N, N), C_ikj(N, N), C_tiled_ikj(N, N);
        Tensor C_u4(N, N), C_t_u4(N, N), C_t_u8(N, N);
        Tensor C_blocked(N, N);
        
        init_random(A, 42);
        init_random(B, 43);
        
        // Zerar C_blocked antes do teste
        std::fill(C_blocked.data.begin(), C_blocked.data.end(), 0.0f);
        
        long long flops = 2LL * N * N * N;
        long long bytes = (long long)N * N * sizeof(float) * 3;
        BenchConfig cfg;
        cfg.warmup_runs = 1;
        cfg.measured_runs = 3;
        cfg.verbose = false;
        
        // Naive (baseline)
        auto r_naive = Benchmark::run("matmul_naive_" + std::to_string(N),
            [&]() { matmul_naive(C_naive, A, B); }, N, flops, bytes, cfg);
        results.push_back(r_naive);
        double baseline_ms = r_naive.median_ns / 1e6;
        
        print_result("naive (baseline)", r_naive.gflops, baseline_ms, baseline_ms, true);
        
        // i-k-j
        auto r_ikj = Benchmark::run("matmul_ikj_" + std::to_string(N),
            [&]() { matmul_ikj(C_ikj, A, B); }, N, flops, bytes, cfg);
        results.push_back(r_ikj);
        print_result("ikj", r_ikj.gflops, r_ikj.median_ns / 1e6, baseline_ms,
                     verify(C_naive, C_ikj));
        
        // tiled + ikj
        auto r_t_ikj = Benchmark::run("matmul_tiled_ikj_" + std::to_string(N),
            [&]() { matmul_tiled_ikj(C_tiled_ikj, A, B, 128); }, N, flops, bytes, cfg);
        results.push_back(r_t_ikj);
        print_result("tiled+ikj T=128", r_t_ikj.gflops, r_t_ikj.median_ns / 1e6, baseline_ms,
                     verify(C_naive, C_tiled_ikj));
        
        // ikj + unroll4
        auto r_u4 = Benchmark::run("matmul_ikj_unroll4_" + std::to_string(N),
            [&]() { matmul_ikj_unroll4(C_u4, A, B); }, N, flops, bytes, cfg);
        results.push_back(r_u4);
        print_result("ikj + unroll4", r_u4.gflops, r_u4.median_ns / 1e6, baseline_ms,
                     verify(C_naive, C_u4));
        
        // tiled + ikj + unroll4
        auto r_t_u4 = Benchmark::run("matmul_tiled_unroll4_" + std::to_string(N),
            [&]() { matmul_tiled_unroll4(C_t_u4, A, B, 128); }, N, flops, bytes, cfg);
        results.push_back(r_t_u4);
        print_result("tiled+unroll4 T=128", r_t_u4.gflops, r_t_u4.median_ns / 1e6, baseline_ms,
                     verify(C_naive, C_t_u4));
        
        // tiled + ikj + unroll8 + register blocking
        auto r_t_u8 = Benchmark::run("matmul_tiled_unroll8_" + std::to_string(N),
            [&]() { matmul_tiled_unroll8(C_t_u8, A, B, 128); }, N, flops, bytes, cfg);
        results.push_back(r_t_u8);
        print_result("tiled+unroll8+reg *", r_t_u8.gflops, r_t_u8.median_ns / 1e6, baseline_ms,
                     verify(C_naive, C_t_u8));
        
        // ========== NOVA IMPLEMENTAÇÃO (AVX2 + Register Blocking + OpenMP) ==========
        auto r_blocked = Benchmark::run("matmul_blocked_avx2_" + std::to_string(N),
            [&]() { matmul_blocked(N, N, N, A.data.data(), B.data.data(), C_blocked.data.data()); },
            N, flops, bytes, cfg);
        results.push_back(r_blocked);
        print_result("matmul_blocked (FMA)", r_blocked.gflops, r_blocked.median_ns / 1e6, baseline_ms,
                     verify(C_naive, C_blocked));
    }
    
    // Tile sweep
    std::cout << "\n--- Tile sweep: tiled_unroll8 em 1024x1024 ---" << std::endl;
    int N = 1024;
    Tensor A(N, N), B(N, N), C(N, N);
    init_random(A, 42);
    init_random(B, 43);
    
    BenchConfig cfg2;
    cfg2.warmup_runs = 1;
    cfg2.measured_runs = 3;
    cfg2.verbose = false;
    
    for (int T : {32, 64, 96, 128, 192, 256}) {
        auto r = Benchmark::run("matmul_unroll8_T" + std::to_string(T),
            [&]() { matmul_tiled_unroll8(C, A, B, T); },
            N, 2LL * N * N * N, 0, cfg2);
        results.push_back(r);
        std::cout << "  T=" << std::setw(4) << T << ":  " 
                  << std::setw(7) << std::fixed << std::setprecision(2) << r.gflops 
                  << " GFLOPS  (" << std::setprecision(1) << (r.median_ns / 1e6) << " ms)" 
                  << std::endl;
    }
    
    Benchmark::save_csv(results, "bench_matmul_optimized.csv");
}