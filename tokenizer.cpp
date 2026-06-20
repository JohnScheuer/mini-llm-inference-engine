#include "tokenizer.h"
#include <iostream>
#include <vector>

Tokenizer::Tokenizer() {}
Tokenizer::~Tokenizer() {}

bool Tokenizer::load(const std::string& path) {
    auto status = sp.Load(path);
    if (!status.ok()) {
        std::cerr << "[SentencePiece] Erro: " << status.ToString() << std::endl;
        return false;
    }
    std::cout << "[Tokenizer] Loaded via SentencePiece (.model)" << std::endl;
    return true;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    sp.Encode(text, &ids);
    return ids;
}

std::string Tokenizer::decode(int prev_token, int token) const {
    std::string s;
    // Criamos o vetor de inteiros de forma explícita
    std::vector<int> ids;
    ids.push_back(token);
    
    // Agora o compilador sabe exatamente qual função chamar
    sp.Decode(ids, &s);
    return s;
}