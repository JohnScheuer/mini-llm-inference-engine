#include "model.h"
#include <iostream>
#include <omp.h>
#include <chrono>

// --------------------------------------------------
// Argmax simples (greedy decoding)
// --------------------------------------------------
int argmax(const Tensor& logits) {
    int max_id = 0;
    float max_val = logits.data[0];

    for (int i = 1; i < logits.cols; i++) {
        if (logits.data[i] > max_val) {
            max_val = logits.data[i];
            max_id = i;
        }
    }
    return max_id;
}

int main() {
    std::cout << "==============================================" << std::endl;
    std::cout << "  Mini-LLM Inference Engine (AVX2 Optimized)" << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "CPU threads disponiveis: "
              << omp_get_max_threads() << std::endl;

    // ✅ Ajuste estes valores conforme seu modelo
    int vocab_size = 32000;
    int dim = 512;
    int hidden_dim = 2048;
    int n_layers = 6;
    int n_heads = 8;
    int max_seq = 256;

    Model model(vocab_size, dim, hidden_dim,
                n_layers, n_heads, max_seq);

    // --------------------------------------------------
    // Imprime configuração real do modelo
    // --------------------------------------------------
    std::cout << "\n[Model Config]\n";
    std::cout << "vocab_size   = " << model.vocab_size << std::endl;
    std::cout << "dim          = " << model.dim << std::endl;
    std::cout << "hidden_dim   = " << model.hidden_dim << std::endl;
    std::cout << "n_layers     = " << model.n_layers << std::endl;
    std::cout << "n_heads      = " << model.n_heads << std::endl;
    std::cout << "max_seq_len  = " << model.max_seq_len << std::endl;
    std::cout << "head_dim     = "
              << model.dim / model.n_heads << std::endl;

    int max_tokens = 128;
    int token = 1;   // token inicial dummy
    int pos = 0;

    Tensor logits(1, model.vocab_size);

    std::cout << "\n[Inference] Gerando "
              << max_tokens << " tokens...\n";

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < max_tokens; i++) {
        model_forward(logits, model, token, pos);
        token = argmax(logits);
        pos++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed =
        std::chrono::duration<double>(end - start).count();

    double tok_per_sec = max_tokens / elapsed;

    std::cout << "\n[Performance]\n";
    std::cout << "Tempo total: " << elapsed << " s\n";
    std::cout << "Tokens/s:    " << tok_per_sec << std::endl;

    return 0;
}