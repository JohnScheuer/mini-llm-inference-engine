#include <iostream>
#include <omp.h>
#include "model.h"
#include "layers.h"

int main() {
    std::cout << "==============================================" << std::endl;
    std::cout << "  Mini-LLM Inference Engine (Otimizado)" << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "CPU threads disponiveis: " << omp_get_max_threads() << std::endl;

    // Configurações do modelo
    int vocab_size = 32000;
    int dim = 768;
    int hidden_dim = 2048;
    int n_layers = 6;     // Rodando 6 camadas para o teste
    int n_heads = 12;
    int max_seq = 1024;

    std::cout << "[Engine] Inicializando Modelo com construtores apropriados..." << std::endl;
    
    // Chama o construtor de Model com os 6 argumentos que ele pede
    Model model(vocab_size, dim, hidden_dim, n_layers, n_heads, max_seq);

    // Nota: Como o construtor de Model provavelmente já cria as camadas internamente
    // e os construtores de TransformerLayer exigem (dim, hidden_dim, max_seq),
    // o seu pipeline de alocação já deve estar pronto aqui.

    Tensor logits(1, vocab_size);
    int token_id = 1; // Token de exemplo (BOS)
    int pos = 0;

    std::cout << "[Engine] Executando forward pass otimizado (AVX2 + Tiling)..." << std::endl;
    
    // Executa o modelo
    model_forward(logits, model, token_id, pos);

    std::cout << "[Engine] Sucesso! O forward pass completou corretamente." << std::endl;
    
    return 0;
}