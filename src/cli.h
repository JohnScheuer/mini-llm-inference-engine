#pragma once
#include <string>
#include <vector>

struct CLIArgs {
    std::string model_path;
    std::string vocab_path;       // NOVO
    std::string merges_path;      // NOVO
    std::string prompt_text;      // MUDOU: era prompt_tokens
    int max_tokens;
    float temperature;
    int top_k;
    bool show_help;
    bool valid;
    
    CLIArgs() 
        : vocab_path("../vocab/vocab.txt"),
          merges_path("../vocab/merges.txt"),
          max_tokens(20),
          temperature(0.8f),
          top_k(0),
          show_help(false),
          valid(false) {}
};

CLIArgs parse_args(int argc, char* argv[]);
void print_help(const char* program_name);