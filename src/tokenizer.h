#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>

class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer(); // Adicionada a declaração do destruidor

    bool load(const std::string& path);
    std::string decode(int prev_token, int token) const;
    int vocab_size() const { return (int)vocab.size(); }

private:
    std::vector<std::string> vocab;
    std::vector<float> vocab_scores;
    int max_token_length;
};

#endif