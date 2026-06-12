#include "tokenizer.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <set>
#include <limits>

Tokenizer::Tokenizer() {}

bool Tokenizer::load(const std::string& vocab_path, const std::string& merges_path) {
    // 1. CARREGA VOCAB.TXT
    // Formato: uma string por linha. Linha N = token ID N.
    std::ifstream vocab_file(vocab_path);
    if (!vocab_file.is_open()) {
        std::cerr << "Erro: nao foi possivel abrir " << vocab_path << std::endl;
        return false;
    }
    
    std::string line;
    int id = 0;
    while (std::getline(vocab_file, line)) {
        // Remove \r se houver (arquivos Windows)
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        
        token_to_id[line] = id;
        id_to_token.push_back(line);
        id++;
    }
    vocab_file.close();
    
    std::cout << "Vocab carregado: " << id_to_token.size() << " tokens" << std::endl;
    
    // 2. CARREGA MERGES.TXT
    // Formato: "a b" por linha, em ordem de prioridade
    std::ifstream merges_file(merges_path);
    if (!merges_file.is_open()) {
        std::cerr << "Erro: nao foi possivel abrir " << merges_path << std::endl;
        return false;
    }
    
    int merge_priority = 0;
    while (std::getline(merges_file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        
        // Pula linhas de cabeçalho (algumas tools começam com #)
        if (line.empty() || line[0] == '#') continue;
        
        // Separa "a b" em duas strings
        size_t space = line.find(' ');
        if (space == std::string::npos) continue;
        
        std::string a = line.substr(0, space);
        std::string b = line.substr(space + 1);
        
        merges[{a, b}] = merge_priority++;
    }
    merges_file.close();
    
    std::cout << "Merges carregados: " << merges.size() << " regras" << std::endl;
    
    return true;
}

// Encoda UMA palavra usando BPE
std::vector<std::string> Tokenizer::bpe_word(const std::string& word) const {
    // 1. Começa dividindo a palavra em caracteres
    std::vector<std::string> tokens;
    for (char c : word) {
        tokens.push_back(std::string(1, c));
    }
    
    // 2. Aplica merges repetidamente
    while (tokens.size() > 1) {
        // Encontra o par com MENOR prioridade (aplicado primeiro)
        int best_priority = std::numeric_limits<int>::max();
        size_t best_idx = 0;
        bool found = false;
        
        for (size_t i = 0; i < tokens.size() - 1; i++) {
            auto pair = std::make_pair(tokens[i], tokens[i + 1]);
            auto it = merges.find(pair);
            if (it != merges.end() && it->second < best_priority) {
                best_priority = it->second;
                best_idx = i;
                found = true;
            }
        }
        
        // Se nenhum par pode ser merged, paramos
        if (!found) break;
        
        // Faz o merge: junta tokens[best_idx] e tokens[best_idx+1]
        std::string merged = tokens[best_idx] + tokens[best_idx + 1];
        tokens[best_idx] = merged;
        tokens.erase(tokens.begin() + best_idx + 1);
    }
    
    return tokens;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> result;
    
    // Pré-tokeniza por espaço (simplificado)
    // BPE real (GPT-2) usa regex complexo; vamos com simples por agora
    std::stringstream ss(text);
    std::string word;
    bool first = true;
    
    while (ss >> word) {
        // No GPT-2/LLaMA, palavras (exceto a primeira) começam com espaço
        // Usamos "Ġ" para representar espaço (convenção do GPT-2)
        // Aqui vamos simplificar: adicionamos espaço no início se não for a primeira palavra
        std::string to_encode = first ? word : (" " + word);
        first = false;
        
        // Aplica BPE
        auto pieces = bpe_word(to_encode);
        
        // Converte pieces em IDs
        for (const auto& piece : pieces) {
            auto it = token_to_id.find(piece);
            if (it != token_to_id.end()) {
                result.push_back(it->second);
            } else {
                // Token desconhecido - poderia ser <unk> mas vamos só avisar
                std::cerr << "Aviso: token desconhecido '" << piece << "'" << std::endl;
            }
        }
    }
    
    return result;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    std::string result;
    for (int id : tokens) {
        if (id >= 0 && id < (int)id_to_token.size()) {
            result += id_to_token[id];
        }
    }
    return result;
}