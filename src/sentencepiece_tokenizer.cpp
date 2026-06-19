#include "sentencepiece_tokenizer.hpp"
#include <fstream>
#include <iostream>
#include <limits>

bool SentencePieceTokenizer::load_model(const std::string& path) {

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Erro ao abrir tokenizer.model\n";
        return false;
    }

    std::string data((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());

    // Parser simplificado (não protobuf completo)
    size_t pos = 0;
    int id = 0;

    while ((pos = data.find("piece", pos)) != std::string::npos) {

        size_t start = data.find('"', pos + 6);
        size_t end = data.find('"', start + 1);

        if (start == std::string::npos || end == std::string::npos)
            break;

        std::string token = data.substr(start + 1, end - start - 1);

        pieces.push_back({token, 0.0f});
        token_to_id[token] = id++;

        pos = end;
    }

    std::cout << "[SentencePiece] Loaded " << pieces.size() << " tokens\n";
    return true;
}