#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>

class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();

    // Carrega o vocabulário do tokenizer.json oficial (Llama/TinyLlama)
    bool load(const std::string& path);
    
    // Codifica texto em IDs usando busca gulosa (Greedy Match)
    std::vector<int> encode(const std::string& text) const;
    
    // Decodifica um ID de token em string, tratando marcadores de espaço e bytes hexadecimais
    std::string decode(int prev_token, int token) const;

private:
    // unordered_map é mais rápido para buscas de strings durante o encode
    std::unordered_map<std::string, int> vocab;
    std::vector<std::string> inv_vocab;
};

#endif