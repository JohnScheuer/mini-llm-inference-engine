#include "tokenizer.h"
#include <iostream>

Tokenizer::Tokenizer() {}
Tokenizer::~Tokenizer() {}

bool Tokenizer::load(const std::string& path) {
    auto status = processor.Load(path);

    if (!status.ok()) {
        std::cerr << "Erro ao carregar tokenizer.model: "
                  << status.ToString() << std::endl;
        return false;
    }

    std::cout << "[Tokenizer] "
              << processor.GetPieceSize()
              << " tokens carregados via SentencePiece."
              << std::endl;

    return true;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    processor.Encode(text, &ids);
    return ids;
}

std::string Tokenizer::decode(int prev_token, int token) const {
    if (token < 0) return "";

    std::string piece = processor.IdToPiece(token);

    if (piece == "▁")
        return " ";

    if (piece.rfind("▁", 0) == 0)
        return " " + piece.substr(3);

    return piece;
}
