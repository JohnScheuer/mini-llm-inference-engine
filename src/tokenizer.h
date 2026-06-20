#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>

class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();
    bool load(const std::string& path);
    std::vector<int> encode(const std::string& text) const;
    std::string decode(int prev_token, int token) const;

private:
    std::unordered_map<std::string, int> vocab;
    std::vector<std::string> inv_vocab;
};

#endif
