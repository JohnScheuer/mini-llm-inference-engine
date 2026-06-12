#include "layers.h"
#include "tensor.h"
#include "transformer.h"
#include "model.h"
#include "generate.h"
#include "weights.h"
#include "cli.h"
#include <iostream>
#include <omp.h>
#include <chrono>

int main(int argc, char* argv[]) {
    // 1. Parse dos argumentos
    CLIArgs args = parse_args(argc, argv);
    
    // 2. Mostrar ajuda se solicitado
    if (args.show_help) {
        print_help(argv[0]);
        return 0;
    }
    
    // 3. Validar argumentos
    if (!args.valid) {
        std::cerr << "\nUse '" << argv[0] << " --help' para ver as opcoes." << std::endl;
        return 1;
    }
    
    // 4. Banner
    std::cout << "========================================" << std::endl;
    std::cout << "  Mini-LLM Inference Engine v1.0" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Threads (OpenMP): " << omp_get_max_threads() << std::endl;
    std::cout << std::endl;
    
    // 5. Carregar modelo
    auto t_start = std::chrono::high_resolution_clock::now();
    Model* model = load_model(args.model_path);
    if (!model) {
        std::cerr << "Falha ao carregar o modelo!" << std::endl;
        return 1;
    }
    auto t_load = std::chrono::high_resolution_clock::now();
    auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_load - t_start).count();
    std::cout << "   Tempo de carregamento: " << load_ms << " ms" << std::endl;
    
    // 6. Mostrar configuracao
    std::cout << "\n--- Configuracao da geracao ---" << std::endl;
    std::cout << "Prompt tokens: ";
    for (int t : args.prompt_tokens) std::cout << t << " ";
    std::cout << "(" << args.prompt_tokens.size() << " tokens)" << std::endl;
    std::cout << "Max tokens: " << args.max_tokens << std::endl;
    std::cout << "Temperature: " << args.temperature << std::endl;
    std::cout << "Top-K: " << (args.top_k > 0 ? std::to_string(args.top_k) : "off") << std::endl;
    
    // 7. Validar que o prompt cabe no modelo
    if (args.prompt_tokens.size() >= (size_t)model->max_seq_len) {
        std::cerr << "Erro: prompt muito longo para o modelo!" << std::endl;
        std::cerr << "  Maximo: " << model->max_seq_len << std::endl;
        delete model;
        return 1;
    }
    
    // Validar IDs do prompt
    for (int t : args.prompt_tokens) {
        if (t < 0 || t >= model->vocab_size) {
            std::cerr << "Erro: token ID invalido (" << t 
                      << "). Deve estar entre 0 e " << (model->vocab_size - 1) << std::endl;
            delete model;
            return 1;
        }
    }
    
    // 8. Geracao
    std::cout << "\n--- Iniciando geracao ---" << std::endl;
    auto t_gen_start = std::chrono::high_resolution_clock::now();
    
    auto tokens = generate(
        *model,
        args.prompt_tokens,
        args.max_tokens,
        args.temperature,
        args.top_k
    );
    
    auto t_gen_end = std::chrono::high_resolution_clock::now();
    auto gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_gen_end - t_gen_start).count();
    
    // 9. Resultado final
    std::cout << "\n--- Resultado final ---" << std::endl;
    std::cout << "Sequencia completa: ";
    for (int t : tokens) std::cout << t << " ";
    std::cout << std::endl;
    
    int generated = tokens.size() - args.prompt_tokens.size();
    float tokens_per_sec = (gen_ms > 0) ? (1000.0f * generated / gen_ms) : 0.0f;
    
    std::cout << "\n--- Estatisticas ---" << std::endl;
    std::cout << "Tokens gerados: " << generated << std::endl;
    std::cout << "Tempo de geracao: " << gen_ms << " ms" << std::endl;
    std::cout << "Velocidade: " << tokens_per_sec << " tokens/s" << std::endl;
    
    // 10. Cleanup
    delete model;
    
    std::cout << "\nGeracao concluida!" << std::endl;
    return 0;
}