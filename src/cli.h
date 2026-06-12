#pragma once
#include <string>
#include <vector>

// Struct que guarda os argumentos parseados da CLI
struct CLIArgs {
    std::string model_path;          // --model / -m
    std::vector<int> prompt_tokens;  // --prompt / -p (formato: "5,12,7")
    int max_tokens;                  // --tokens / -t
    float temperature;               // --temperature
    int top_k;                       // --top-k
    bool show_help;                  // --help / -h
    bool valid;                      // se os args estão completos
    
    // Construtor com valores padrão
    CLIArgs() 
        : max_tokens(20),
          temperature(0.8f),
          top_k(0),
          show_help(false),
          valid(false) {}
};

// Faz o parse dos argumentos do main(argc, argv)
CLIArgs parse_args(int argc, char* argv[]);

// Mostra ajuda da CLI
void print_help(const char* program_name);