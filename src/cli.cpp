#include "cli.h"
#include <iostream>
#include <sstream>
#include <cstring>

void print_help(const char* program_name) {
    std::cout << "Mini-LLM Inference Engine\n\n";
    std::cout << "Uso: " << program_name << " [opcoes]\n\n";
    std::cout << "Opcoes obrigatorias:\n";
    std::cout << "  -m, --model <arquivo>      Caminho do modelo .bin\n";
    std::cout << "  -p, --prompt <ids>         Tokens do prompt (ex: \"5,12,7\")\n\n";
    std::cout << "Opcoes opcionais:\n";
    std::cout << "  -t, --tokens <n>           Numero de tokens a gerar (padrao: 20)\n";
    std::cout << "      --temperature <f>      Temperatura de sampling (padrao: 0.8)\n";
    std::cout << "                             0.0 = argmax (deterministico)\n";
    std::cout << "      --top-k <n>            Top-K sampling (padrao: 0 = desligado)\n";
    std::cout << "  -h, --help                 Mostra esta ajuda\n\n";
    std::cout << "Exemplos:\n";
    std::cout << "  " << program_name << " -m model.bin -p \"5,12,7\" -t 30\n";
    std::cout << "  " << program_name << " --model model.bin --prompt \"5,12,7\" \\\n";
    std::cout << "      --tokens 50 --temperature 0.5 --top-k 10\n";
}

// Parser auxiliar: converte "5,12,7" em vector<int>{5, 12, 7}
static std::vector<int> parse_token_list(const std::string& str) {
    std::vector<int> tokens;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        try {
            tokens.push_back(std::stoi(item));
        } catch (...) {
            std::cerr << "Erro: token invalido '" << item << "'" << std::endl;
            return {};
        }
    }
    return tokens;
}

CLIArgs parse_args(int argc, char* argv[]) {
    CLIArgs args;
    
    if (argc == 1) {
        args.show_help = true;
        return args;
    }
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        // --help / -h
        if (arg == "-h" || arg == "--help") {
            args.show_help = true;
            return args;
        }
        
        // --model / -m
        else if (arg == "-m" || arg == "--model") {
            if (i + 1 >= argc) {
                std::cerr << "Erro: " << arg << " requer um argumento" << std::endl;
                return args;
            }
            args.model_path = argv[++i];
        }
        
        // --prompt / -p
        else if (arg == "-p" || arg == "--prompt") {
            if (i + 1 >= argc) {
                std::cerr << "Erro: " << arg << " requer um argumento" << std::endl;
                return args;
            }
            args.prompt_tokens = parse_token_list(argv[++i]);
            if (args.prompt_tokens.empty()) {
                std::cerr << "Erro: prompt vazio ou invalido" << std::endl;
                return args;
            }
        }
        
        // --tokens / -t
        else if (arg == "-t" || arg == "--tokens") {
            if (i + 1 >= argc) {
                std::cerr << "Erro: " << arg << " requer um argumento" << std::endl;
                return args;
            }
            try {
                args.max_tokens = std::stoi(argv[++i]);
            } catch (...) {
                std::cerr << "Erro: numero de tokens invalido" << std::endl;
                return args;
            }
        }
        
        // --temperature
        else if (arg == "--temperature") {
            if (i + 1 >= argc) {
                std::cerr << "Erro: " << arg << " requer um argumento" << std::endl;
                return args;
            }
            try {
                args.temperature = std::stof(argv[++i]);
            } catch (...) {
                std::cerr << "Erro: temperatura invalida" << std::endl;
                return args;
            }
        }
        
        // --top-k
        else if (arg == "--top-k") {
            if (i + 1 >= argc) {
                std::cerr << "Erro: " << arg << " requer um argumento" << std::endl;
                return args;
            }
            try {
                args.top_k = std::stoi(argv[++i]);
            } catch (...) {
                std::cerr << "Erro: top-k invalido" << std::endl;
                return args;
            }
        }
        
        // Argumento desconhecido
        else {
            std::cerr << "Erro: argumento desconhecido '" << arg << "'" << std::endl;
            std::cerr << "Use --help para ver as opcoes disponiveis." << std::endl;
            return args;
        }
    }
    
    // Validar argumentos obrigatórios
    if (args.model_path.empty()) {
        std::cerr << "Erro: --model e obrigatorio" << std::endl;
        return args;
    }
    if (args.prompt_tokens.empty()) {
        std::cerr << "Erro: --prompt e obrigatorio" << std::endl;
        return args;
    }
    
    args.valid = true;
    return args;
}