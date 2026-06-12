#include "benchmark.h"
#include <cmath>
#include <iomanip>
#include <omp.h>

BenchResult Benchmark::run(
    const std::string& name,
    std::function<void()> func,
    int problem_size,
    long long flop_count,
    long long bytes_accessed,
    BenchConfig config
) {
    using clock = std::chrono::high_resolution_clock;
    
    BenchResult result;
    result.name = name;
    result.problem_size = problem_size;
    result.num_threads = omp_get_max_threads();
    result.warmup_runs = config.warmup_runs;
    result.measured_runs = config.measured_runs;
    
    // === WARMUP ===
    // Aquece o cache, o branch predictor, e tira a CPU do power saving
    if (config.verbose) {
        std::cout << "  [" << name << "] Warmup (" << config.warmup_runs << " runs)..." 
                  << std::flush;
    }
    for (int i = 0; i < config.warmup_runs; i++) {
        func();
    }
    if (config.verbose) std::cout << " ok" << std::endl;
    
    // === MEDIDAS ===
    std::vector<double> times_ns;
    times_ns.reserve(config.measured_runs);
    
    if (config.verbose) {
        std::cout << "  [" << name << "] Medindo (" << config.measured_runs << " runs)..." 
                  << std::flush;
    }
    
    for (int i = 0; i < config.measured_runs; i++) {
        auto start = clock::now();
        func();
        auto end = clock::now();
        
        double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        times_ns.push_back(ns);
    }
    if (config.verbose) std::cout << " ok" << std::endl;
    
    // === ESTATÍSTICAS ===
    // Ordena pra pegar mediana
    std::sort(times_ns.begin(), times_ns.end());
    result.median_ns = times_ns[times_ns.size() / 2];
    result.min_ns = times_ns.front();
    result.max_ns = times_ns.back();
    
    // Desvio padrão
    double mean = 0.0;
    for (double t : times_ns) mean += t;
    mean /= times_ns.size();
    
    double variance = 0.0;
    for (double t : times_ns) variance += (t - mean) * (t - mean);
    variance /= times_ns.size();
    result.std_dev_ns = std::sqrt(variance);
    
    // GFLOPS = flops / seconds / 1e9
    if (flop_count > 0) {
        double seconds = result.median_ns / 1e9;
        result.gflops = flop_count / seconds / 1e9;
    } else {
        result.gflops = 0.0;
    }
    
    // Bandwidth = bytes / seconds / 1e9
    if (bytes_accessed > 0) {
        double seconds = result.median_ns / 1e9;
        result.bandwidth_gbs = bytes_accessed / seconds / 1e9;
    } else {
        result.bandwidth_gbs = 0.0;
    }
    
    return result;
}

void Benchmark::print(const BenchResult& r) {
    std::cout << "\n┌─ " << r.name << " ─\n";
    std::cout << "│  Tamanho do problema: " << r.problem_size << "\n";
    std::cout << "│  Threads:             " << r.num_threads << "\n";
    std::cout << "│  Runs (warmup+med):   " << r.warmup_runs << " + " << r.measured_runs << "\n";
    std::cout << "│  Tempo mediano:       " << std::fixed << std::setprecision(2) 
              << (r.median_ns / 1e6) << " ms\n";
    std::cout << "│  Min / Max:           " << (r.min_ns / 1e6) << " / " 
              << (r.max_ns / 1e6) << " ms\n";
    std::cout << "│  Desvio padrão:       " << (r.std_dev_ns / 1e6) << " ms\n";
    if (r.gflops > 0) {
        std::cout << "│  Performance:         " << std::setprecision(2) << r.gflops << " GFLOPS\n";
    }
    if (r.bandwidth_gbs > 0) {
        std::cout << "│  Bandwidth:           " << r.bandwidth_gbs << " GB/s\n";
    }
    std::cout << "└─\n";
}

void Benchmark::print_table(const std::vector<BenchResult>& results) {
    if (results.empty()) return;
    
    std::cout << "\n";
    std::cout << std::left
              << std::setw(35) << "Benchmark"
              << std::setw(10) << "Size"
              << std::setw(10) << "Threads"
              << std::setw(15) << "Tempo (ms)"
              << std::setw(12) << "GFLOPS"
              << std::setw(12) << "GB/s"
              << "\n";
    std::cout << std::string(94, '-') << "\n";
    
    for (const auto& r : results) {
        std::cout << std::left
                  << std::setw(35) << r.name
                  << std::setw(10) << r.problem_size
                  << std::setw(10) << r.num_threads
                  << std::setw(15) << std::fixed << std::setprecision(3) << (r.median_ns / 1e6)
                  << std::setw(12) << std::setprecision(2) << r.gflops
                  << std::setw(12) << r.bandwidth_gbs
                  << "\n";
    }
    std::cout << "\n";
}

void Benchmark::save_csv(
    const std::vector<BenchResult>& results,
    const std::string& filepath
) {
    std::ofstream f(filepath);
    if (!f.is_open()) {
        std::cerr << "Erro ao criar CSV: " << filepath << std::endl;
        return;
    }
    
    // Header
    f << "name,problem_size,num_threads,median_ns,min_ns,max_ns,std_dev_ns,gflops,bandwidth_gbs\n";
    
    // Linhas
    for (const auto& r : results) {
        f << r.name << ","
          << r.problem_size << ","
          << r.num_threads << ","
          << r.median_ns << ","
          << r.min_ns << ","
          << r.max_ns << ","
          << r.std_dev_ns << ","
          << r.gflops << ","
          << r.bandwidth_gbs << "\n";
    }
    
    f.close();
    std::cout << "📊 CSV salvo: " << filepath << std::endl;
}