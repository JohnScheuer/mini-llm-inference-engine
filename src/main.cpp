#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <omp.h>
#include "model.h"
#include "tokenizer.h"
#include "layers.h"

bool load_model_weights(Model& model, const std::string& path);

int argmax(const Tensor& logits) {
    // Busca do token 3 para cima para ignorar bytes nulos de escape
    int best_id = 3;
    float max_p = logits.data[3];
    for (int i = 4; i < (int)logits.data.size(); i++) {
        if (logits.data[i] > max_p) {
            max_p = logits.data[i];
            best_id = i;
        }
    }
    return best_id;
}

int main() {
    Tokenizer tokenizer;
    if (!tokenizer.load("vocab/tokenizer.bin")) return 1;

    Model model(32000, 1, 1, 1, 1, 1);
    if (!load_model_weights(model, "model.bin")) return 1;

    std::cout << "==============================================" << std::endl;
    std::cout << "  Mini-LLM Engine - Geração Stories15M" << std::endl;
    std::cout << "==============================================" << std::endl;

    Tensor logits(1, model.vocab_size);
    int token = 1; // BOS
    int pos = 0;
    std::cout << "\n[História]: " << std::flush;

    auto start = std::chrono::high_resolution_clock::now();

    for (pos = 0; pos < 100; pos++) {
        model_forward(logits, model, token, pos);
        
        int next_token = argmax(logits);
        
        std::cout << tokenizer.decode(token, next_token) << std::flush;
        
        token = next_token;
        if (token == 2) break; // EOS
    }

    auto end = std::chrono::high_resolution_clock::now();
    double s = std::chrono::duration<double>(end - start).count();
    std::cout << "\n\n----------------------------------------------\n";
    std::cout << "[Performance] " << (double)pos / s << " tokens/s" << std::endl;

    return 0;
}