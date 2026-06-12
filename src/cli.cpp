#include "cli.h"
#include <iostream>
#include <cstring>

void print_help(const char* program_name) {
    std::cout << "Mini-LLM Inference Engine\n\n";
    std::cout << "Uso: " << program_name << " [opcoes]\n\n";
    std::cout << "Opcoes obrigatorias:\n";
    std::cout << "  -m, --model <arquivo>      Caminho do modelo .bin\n";
    std::cout << "  -p, --prompt <texto>       Texto do prompt\n\n";
    std::cout << "Opcoes do tokenizer:\n";
    std::cout << "      --vocab <arquivo>      Caminho do vocab.txt (padrao: ../vocab/vocab.txt)\n";
    std::cout << "      --merges <arquivo>     Caminho do merges.txt (padrao: ../vocab/merges.txt)\n\n";
    std::cout << "Opcoes de geracao:\n";
    std::cout << "  -t, --tokens <n>           Numero de tokens a gerar (padrao: 20)\n";
    std::cout << "      --temperature <f>      Temperatura (padrao: 0.8, 0 = argmax)\n";
    std::cout << "      --top-k <n>            Top-K (padrao: 0 = desligado)\n";
    std::cout << "  -h, --help                 Mostra esta ajuda\n\n";
    std::cout << "Exemplos:\n";
    std::cout << "  " << program_name << " -m model.bin -p \"hello world\" -t 30\n";
}

CLIArgs parse_args(int argc, char* argv[]) {
    CLIArgs args;
    
    if (argc == 1) {
        args.show_help = true;
        return args;
    }
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            args.show_help = true;
            return args;
        }
        else if (arg == "-m" || arg == "--model") {
            if (i + 1 >= argc) { std::cerr << "Erro: " << arg << " requer um argumento\n"; return args; }
            args.model_path = argv[++i];
        }
        else if (arg == "-p" || arg == "--prompt") {
            if (i + 1 >= argc) { std::cerr << "Erro: " << arg << " requer um argumento\n"; return args; }
            args.prompt_text = argv[++i];
        }
        else if (arg == "--vocab") {
            if (i + 1 >= argc) { std::cerr << "Erro: " << arg << " requer um argumento\n"; return args; }
            args.vocab_path = argv[++i];
        }
        else if (arg == "--merges") {
            if (i + 1 >= argc) { std::cerr << "Erro: " << arg << " requer um argumento\n"; return args; }
            args.merges_path = argv[++i];
        }
        else if (arg == "-t" || arg == "--tokens") {
            if (i + 1 >= argc) { std::cerr << "Erro: " << arg << " requer um argumento\n"; return args; }
            try { args.max_tokens = std::stoi(argv[++i]); } 
            catch (...) { std::cerr << "Erro: numero invalido\n"; return args; }
        }
        else if (arg == "--temperature") {
            if (i + 1 >= argc) { std::cerr << "Erro: " << arg << " requer um argumento\n"; return args; }
            try { args.temperature = std::stof(argv[++i]); } 
            catch (...) { std::cerr << "Erro: temperatura invalida\n"; return args; }
        }
        else if (arg == "--top-k") {
            if (i + 1 >= argc) { std::cerr << "Erro: " << arg << " requer um argumento\n"; return args; }
            try { args.top_k = std::stoi(argv[++i]); } 
            catch (...) { std::cerr << "Erro: top-k invalido\n"; return args; }
        }
        else {
            std::cerr << "Erro: argumento desconhecido '" << arg << "'\n";
            return args;
        }
    }
    
    if (args.model_path.empty()) { std::cerr << "Erro: --model obrigatorio\n"; return args; }
    if (args.prompt_text.empty()) { std::cerr << "Erro: --prompt obrigatorio\n"; return args; }
    
    args.valid = true;
    return args;
}