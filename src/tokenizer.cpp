#include "tokenizer.h"
#include "json.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <iomanip>

using json = nlohmann::json;

Tokenizer::Tokenizer() {}
Tokenizer::~Tokenizer() {}

bool Tokenizer::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) return false;
    json data = json::parse(f);
    json v_json = data.contains("model") ? data["model"]["vocab"] : data["vocab"];
    
    vocab.clear();
    int max_id = 0;
    for (auto& el : v_json.items()) {
        vocab[el.key()] = el.value();
        max_id = std::max(max_id, (int)el.value());
    }
    inv_vocab.assign(max_id + 1, "");
    for (auto& el : vocab) inv_vocab[el.second] = el.first;
    
    std::cout << "[Tokenizer] " << vocab.size() << " tokens loaded via Native BPE." << std::endl;
    return true;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    if (text.empty()) return ids;
    std::string p = "\xe2\x96\x81"; 
    for (char c : text) {
        if (c == ' ') p += "\xe2\x96\x81";
        else p += c;
    }
    size_t start = 0;
    while (start < p.length()) {
        int best_id = -1; size_t best_len = 0;
        for (size_t len = std::min(p.length() - start, (size_t)32); len > 0; len--) {
            auto it = vocab.find(p.substr(start, len));
            if (it != vocab.end()) { best_id = it->second; best_len = len; break; }
        }
        if (best_id != -1) { ids.push_back(best_id); start += best_len; }
        else {
            unsigned char c = (unsigned char)p[start];
            std::stringstream ss;
            ss << "<0x" << std::uppercase << std::setfill('0') << std::setw(2) << std::hex << (int)c << ">";
            auto it = vocab.find(ss.str());
            if (it != vocab.end()) ids.push_back(it->second);
            start++;
        }
    }
    return ids;
}

std::string Tokenizer::decode(int prev_token, int token) const {
    if (token < 0 || token >= (int)inv_vocab.size()) return "";
    std::string s = inv_vocab[token];
    if (s.length() == 6 && s.substr(0, 3) == "<0x" && s.back() == '>') {
        try {
            return std::string(1, (char)std::stoi(s.substr(3, 2), nullptr, 16));
        } catch(...) { return ""; }
    }
    size_t pos;
    while ((pos = s.find("\xe2\x96\x81")) != std::string::npos) s.replace(pos, 3, " ");
    return s;
}
