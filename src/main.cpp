#include "benchmark/benchmark.h"
#include <iostream>
#include <omp.h>
#include "matmul_blocked.h"

// Declaração externa (definida em bench_matmul.cpp)
void run_matmul_benchmarks();

int main() {
    std::cout << "==============================================" << std::endl;
    std::cout << "  Mini-LLM Benchmark Suite" << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "CPU threads disponiveis: " << omp_get_max_threads() << std::endl;
    
    run_matmul_benchmarks();
    
    return 0;
}