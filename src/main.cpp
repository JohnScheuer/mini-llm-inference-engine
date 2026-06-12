#include "layers.h"
#include "tensor.h"
#include "transformer.h"
#include "model.h"
#include "generate.h"
#include "weights.h"
#include "quantize.h"
#include <iostream>
#include <omp.h>

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

void testFP16Quantization() {
    std::cout << "\n=== Teste Quantizacao FP16 ===" << std::endl;
    
    // Teste 1: Conversão de números específicos
    std::cout << "\n[1] Teste de conversao FP32 <-> FP16:" << std::endl;
    float test_values[] = {1.0f, 0.5f, -0.25f, 3.14159f, 0.0f, -1.0f};
    for (float v : test_values) {
        uint16_t h = fp32_to_fp16(v);
        float back = fp16_to_fp32(h);
        std::cout << "  " << v << " -> 0x" << std::hex << h 
                  << " -> " << std::dec << back 
                  << " (erro: " << (v - back) << ")" << std::endl;
    }
    
    // Teste 2: Salvar e carregar modelo
    int vocab_size = 32;
    int dim = 16;
    int hidden_dim = 32;
    int n_layers = 2;
    int n_heads = 2;
    int max_seq_len = 64;
    
    std::cout << "\n[2] Criando modelo..." << std::endl;
    Model original(vocab_size, dim, hidden_dim, n_layers, n_heads, max_seq_len);
    fill_random_weights(original);
    
    std::cout << "\n[3] Salvando em FP32..." << std::endl;
    save_model(original, "model_fp32.bin", QUANT_FP32);
    
    std::cout << "\n[4] Salvando em FP16..." << std::endl;
    save_model(original, "model_fp16.bin", QUANT_FP16);
    
    std::cout << "\n[5] Carregando FP16 e gerando..." << std::endl;
    Model* loaded = load_model("model_fp16.bin");
    if (!loaded) return;
    
    std::vector<int> prompt = {5, 12, 7};
    
    std::cout << "\n--- Original (FP32) ---" << std::endl;
    auto tokens_orig = generate(original, prompt, 8, 0.0f);
    
    std::cout << "\n--- Carregado (FP16) ---" << std::endl;
    auto tokens_fp16 = generate(*loaded, prompt, 8, 0.0f);
    
    std::cout << "\n[6] Comparacao:" << std::endl;
    bool match = (tokens_orig == tokens_fp16);
    if (match) {
        std::cout << "Tokens IDENTICOS! FP16 manteve a precisao suficiente." << std::endl;
    } else {
        std::cout << "Tokens diferentes (esperado em alguns casos com FP16)." << std::endl;
        std::cout << "  FP32: ";
        for (int t : tokens_orig) std::cout << t << " ";
        std::cout << std::endl << "  FP16: ";
        for (int t : tokens_fp16) std::cout << t << " ";
        std::cout << std::endl;
    }
    
    delete loaded;
}

int main() {
    std::cout << "=== Mini-LLM Inference Engine ===" << std::endl;
    std::cout << "OpenMP ativo. Nucleos: " << omp_get_max_threads() << std::endl;
    
    testFP16Quantization();
    
    return 0;
}