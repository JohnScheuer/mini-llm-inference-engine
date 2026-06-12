#include "layers.h"
#include "tensor.h"
#include "transformer.h"
#include "model.h"
#include "generate.h"
#include "weights.h"
#include <iostream>
#include <omp.h>

// Helper: preenche um modelo com pesos aleatórios determinísticos
void fill_random_weights(Model& model) {
    int vocab_size = model.vocab_size;
    int dim = model.dim;
    int hidden_dim = model.hidden_dim;
    int n_layers = model.n_layers;
    
    for (int i = 0; i < vocab_size * dim; i++) {
        model.token_embedding.data[i] = 0.02f * ((i % 13) - 6);
    }
    for (int i = 0; i < dim; i++) {
        model.norm_final.data[i] = 1.0f;
    }
    for (int i = 0; i < dim * vocab_size; i++) {
        model.lm_head.data[i] = 0.02f * ((i % 11) - 5);
    }
    for (int l = 0; l < n_layers; l++) {
        auto& layer = model.layers[l];
        for (int i = 0; i < dim * dim; i++) {
            layer.wq.data[i] = 0.02f * ((i + l) % 7 - 3);
            layer.wk.data[i] = 0.02f * ((i + l) % 5 - 2);
            layer.wv.data[i] = 0.02f * ((i + l) % 3 - 1);
            layer.wo.data[i] = 0.02f * ((i + l) % 4 - 1);
        }
        for (int i = 0; i < dim * hidden_dim; i++) {
            layer.w1.data[i] = 0.02f * ((i + l) % 7 - 3);
            layer.w3.data[i] = 0.02f * ((i + l) % 5 - 2);
        }
        for (int i = 0; i < hidden_dim * dim; i++) {
            layer.w2.data[i] = 0.02f * ((i + l) % 4 - 1);
        }
        for (int i = 0; i < dim; i++) {
            layer.norm_attn.data[i] = 1.0f;
            layer.norm_ffn.data[i] = 1.0f;
        }
    }
}

void testSaveLoad() {
    std::cout << "\n=== Teste Save/Load ===" << std::endl;
    
    int vocab_size = 32;
    int dim = 16;
    int hidden_dim = 32;
    int n_layers = 2;
    int n_heads = 2;
    int max_seq_len = 64;
    
    // ETAPA 1: Cria modelo, preenche pesos, salva
    std::cout << "\n[1] Criando modelo original..." << std::endl;
    Model original(vocab_size, dim, hidden_dim, n_layers, n_heads, max_seq_len);
    fill_random_weights(original);
    
    std::cout << "\n[2] Salvando modelo no disco..." << std::endl;
    save_model(original, "model.bin");
    
    // ETAPA 2: Gera tokens com o modelo original
    std::cout << "\n[3] Gerando com modelo ORIGINAL..." << std::endl;
    std::vector<int> prompt = {5, 12, 7};
    auto tokens_original = generate(original, prompt, 8, 0.0f);  // argmax
    
    // ETAPA 3: Carrega o modelo do disco
    std::cout << "\n[4] Carregando modelo do disco..." << std::endl;
    Model* loaded = load_model("model.bin");
    if (!loaded) {
        std::cerr << "Erro ao carregar!" << std::endl;
        return;
    }
    
    // ETAPA 4: Gera tokens com o modelo carregado
    std::cout << "\n[5] Gerando com modelo CARREGADO..." << std::endl;
    auto tokens_loaded = generate(*loaded, prompt, 8, 0.0f);  // argmax
    
    // ETAPA 5: Compara
    std::cout << "\n[6] Comparando resultados..." << std::endl;
    bool match = (tokens_original == tokens_loaded);
    if (match) {
        std::cout << "✅ Tokens IDENTICOS! Save/Load funcionou!" << std::endl;
    } else {
        std::cout << "❌ Tokens DIFERENTES! Algo deu errado." << std::endl;
        std::cout << "Original: ";
        for (int t : tokens_original) std::cout << t << " ";
        std::cout << std::endl << "Carregado: ";
        for (int t : tokens_loaded) std::cout << t << " ";
        std::cout << std::endl;
    }
    
    delete loaded;  // libera memória
}

int main() {
    std::cout << "=== Mini-LLM Inference Engine ===" << std::endl;
    std::cout << "OpenMP ativo. Nucleos: " << omp_get_max_threads() << std::endl;
    
    testSaveLoad();
    
    return 0;
}