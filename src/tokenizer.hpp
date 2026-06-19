#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <string>
#include <vector>
#include <map>
#include "json.hpp"

class Tokenizer {
public:
    std::map<std::string, int> vocab;
    std::map<int, std::string> inv_vocab;
    
    Tokenizer(const std::string& json_path);
    std::vector<int> encode(std::string text);
    std::string decode(const std::vector<int>& ids);
    
private:
    void replace_all(std::string& str, const std::string& from, const std::string& to);
};

#endif