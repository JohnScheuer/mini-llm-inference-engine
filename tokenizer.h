#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include <sentencepiece_processor.h>

class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();

    bool load(const std::string& path);
    std::vector<int> encode(const std::string& text) const;
    std::string decode(int prev_token, int token) const;

private:
    sentencepiece::SentencePieceProcessor processor;
};

#endif
