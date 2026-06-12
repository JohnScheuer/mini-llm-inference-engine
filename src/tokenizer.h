#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

// Hash function para par de strings (usado no merges)
struct PairHash {
    size_t operator()(const std::pair<std::string, std::string>& p) const {
        return std::hash<std::string>()(p.first) ^ 
               (std::hash<std::string>()(p.second) << 1);
    }
};

class Tokenizer {
public:
    Tokenizer();
    
    // Carrega vocab.txt e merges.txt
    bool load(const std::string& vocab_path, const std::string& merges_path);
    
    // Converte texto em lista de token IDs
    std::vector<int> encode(const std::string& text) const;
    
    // Converte lista de IDs em texto
    std::string decode(const std::vector<int>& tokens) const;
    
    // Tamanho do vocabulário
    int vocab_size() const { return id_to_token.size(); }
    
private:
    // Vocabulário: token (string) → ID (int)
    std::unordered_map<std::string, int> token_to_id;
    
    // Vocabulário inverso: ID → token
    std::vector<std::string> id_to_token;
    
    // Regras de merge: (a, b) → prioridade (menor = aplicado primeiro)
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> merges;
    
    // Encoda UMA palavra com BPE
    std::vector<std::string> bpe_word(const std::string& word) const;
};