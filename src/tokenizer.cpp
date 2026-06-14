#include "tokenizer.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>

Tokenizer::Tokenizer() : max_token_length(0) {}

Tokenizer::~Tokenizer() {} // Agora o compilador vai aceitar

bool Tokenizer::load(const std::string& path) {
    FILE* file = fopen(path.c_str(), "rb");
    if (!file) {
        std::cerr << "[Tokenizer] Erro ao abrir: " << path << std::endl;
        return false;
    }

    if (fread(&max_token_length, sizeof(int), 1, file) != 1) {
        fclose(file);
        return false;
    }

    int expected_vocab_size = 32000;
    vocab.resize(expected_vocab_size);
    vocab_scores.resize(expected_vocab_size);

    for (int i = 0; i < expected_vocab_size; i++) {
        if (fread(&vocab_scores[i], sizeof(float), 1, file) != 1) break;
        
        int len;
        if (fread(&len, sizeof(int), 1, file) != 1) break;
        
        std::vector<char> buf(len + 1);
        if (fread(buf.data(), 1, len, file) != (size_t)len) break;
        buf[len] = '\0';
        
        vocab[i] = std::string(buf.data());
    }

    fclose(file);
    std::cout << "[Tokenizer] 32.000 tokens carregados." << std::endl;
    return true;
}

std::string Tokenizer::decode(int prev_token, int token) const {
    if (token < 0 || token >= (int)vocab.size()) return "";
    
    std::string piece = vocab[token];

    // Converte a string literal "<0x0A>" em uma quebra de linha real
    if (piece == "<0x0A>") return "\n";

    // Tratamento de espaços do Llama (U+2581 ou espaço inicial)
    if (prev_token == 1 && piece[0] == ' ') return piece.substr(1);

    return piece;
}