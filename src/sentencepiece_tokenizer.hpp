#pragma once
#include <vector>
#include <string>
#include <unordered_map>

struct Piece {
    std::string token;
    float score;
};

class SentencePieceTokenizer {
public:
    bool load_model(const std::string& path);
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;

private:
    std::vector<Piece> pieces;
    std::unordered_map<std::string, int> token_to_id;
};