#include "layers.h"
#include "tensor.h"
#include "transformer.h"
#include "model.h"
#include "generate.h"
#include "weights.h"
#include "cli.h"
#include "tokenizer.h"
#include <iostream>
#include <omp.h>
#include <chrono>

int main(int argc, char* argv[]) {
    CLIArgs args = parse_args(argc, argv);
    
    if (args.show_help) {
        print_help(argv[0]);
        return 0;
    }
    
    if (!args.valid) {
        std::cerr << "\nUse '" << argv[0] << " --help' para ver as opcoes." << std::endl;
        return 1;
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "  Mini-LLM Inference Engine v1.0" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Threads (OpenMP): " << omp_get_max_threads() << std::endl;
    std::cout << std::endl;
    
    // 1. CARREGA O TOKENIZER
    std::cout << "[1/3] Carregando tokenizer..." << std::endl;
    Tokenizer tokenizer;
    if (!tokenizer.load(args.vocab_path, args.merges_path)) {
        std::cerr << "Falha ao carregar tokenizer!" << std::endl;
        return 1;
    }
    
    // 2. TOKENIZA O PROMPT
    std::cout << "\n[2/3] Tokenizando prompt: \"" << args.prompt_text << "\"" << std::endl;
    auto prompt_tokens = tokenizer.encode(args.prompt_text);
    std::cout << "   Token IDs: ";
    for (int t : prompt_tokens) std::cout << t << " ";
    std::cout << "(" << prompt_tokens.size() << " tokens)" << std::endl;
    
    if (prompt_tokens.empty()) {
        std::cerr << "Erro: prompt resultou em 0 tokens!" << std::endl;
        return 1;
    }
    
    // 3. CARREGA O MODELO
    std::cout << "\n[3/3] Carregando modelo..." << std::endl;
    auto t_start = std::chrono::high_resolution_clock::now();
    Model* model = load_model(args.model_path);
    if (!model) {
        std::cerr << "Falha ao carregar modelo!" << std::endl;
        return 1;
    }
    auto t_load = std::chrono::high_resolution_clock::now();
    auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_load - t_start).count();
    std::cout << "   Tempo: " << load_ms << " ms" << std::endl;
    
    // Validações
    if (tokenizer.vocab_size() != model->vocab_size) {
        std::cerr << "AVISO: vocab do tokenizer (" << tokenizer.vocab_size() 
                  << ") != vocab do modelo (" << model->vocab_size << ")" << std::endl;
    }
    
    if ((int)prompt_tokens.size() >= model->max_seq_len) {
        std::cerr << "Erro: prompt muito longo" << std::endl;
        delete model;
        return 1;
    }
    
    for (int t : prompt_tokens) {
        if (t >= model->vocab_size) {
            std::cerr << "Erro: token ID " << t << " fora do vocab do modelo" << std::endl;
            delete model;
            return 1;
        }
    }
    
    // 4. GERA
    std::cout << "\n--- Configuracao ---" << std::endl;
    std::cout << "Max tokens: " << args.max_tokens << std::endl;
    std::cout << "Temperature: " << args.temperature << std::endl;
    std::cout << "Top-K: " << (args.top_k > 0 ? std::to_string(args.top_k) : "off") << std::endl;
    
    std::cout << "\n--- Geracao ---" << std::endl;
    auto t_gen_start = std::chrono::high_resolution_clock::now();
    
    auto all_tokens = generate(
        *model,
        prompt_tokens,
        args.max_tokens,
        args.temperature,
        args.top_k
    );
    
    auto t_gen_end = std::chrono::high_resolution_clock::now();
    auto gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_gen_end - t_gen_start).count();
    
    // 5. DECODIFICA
    std::string output_text = tokenizer.decode(all_tokens);
    
    std::cout << "\n--- Resultado ---" << std::endl;
    std::cout << "Prompt original: \"" << args.prompt_text << "\"" << std::endl;
    std::cout << "Texto gerado:    \"" << output_text << "\"" << std::endl;
    
    int generated = all_tokens.size() - prompt_tokens.size();
    float tokens_per_sec = (gen_ms > 0) ? (1000.0f * generated / gen_ms) : 0.0f;
    
    std::cout << "\n--- Estatisticas ---" << std::endl;
    std::cout << "Tokens gerados: " << generated << std::endl;
    std::cout << "Tempo: " << gen_ms << " ms" << std::endl;
    std::cout << "Velocidade: " << tokens_per_sec << " tok/s" << std::endl;
    
    delete model;
    
    std::cout << "\nConcluido!" << std::endl;
    return 0;
}