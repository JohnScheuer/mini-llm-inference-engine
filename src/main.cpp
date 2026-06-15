#include <iostream>
#include <chrono>
#include <vector>
#include "model.h"
#include "tokenizer.h"
#include "layers.h"

bool load_model_weights(Model& model, const std::string& path);

int argmax(const Tensor& logits) {
    int next_token = 3;
    float max_p = -1e20f;
    for (int i = 3; i < (int)logits.data.size(); i++) {
        if (logits.data[i] > max_p) { max_p = logits.data[i]; next_token = i; }
    }
    return next_token;
}

int main() {
    std::cout << "\n>>> HPC ENGINE: 2x UNROLLED INT8 - FINAL STABLE <<<\n" << std::endl;
    Tokenizer tokenizer;
    if (!tokenizer.load("vocab/tokenizer.bin")) return 1;
    Model model(32000, 288, 768, 6, 6, 256);
    if (!load_model_weights(model, "model.bin")) return 1;
    quantize_model_weights(model);
    Tensor logits(1, model.vocab_size);
    int token = 3; int pos = 0;
    std::cout << "[Geração]: " << std::flush;
    auto start = std::chrono::high_resolution_clock::now();
    for (pos = 0; pos < 50; pos++) {
        model_forward(logits, model, token, pos);
        int next_token = argmax(logits);
        std::cout << next_token << " " << std::flush;
        token = next_token;
        if (token == 2) break;
    }
    auto end = std::chrono::high_resolution_clock::now();
    double s = std::chrono::duration<double>(end - start).count();
    std::cout << "\n\n----------------------------------------------\n";
    std::cout << "[Performance] " << (double)pos / s << " tokens/s" << std::endl;
    return 0;
}