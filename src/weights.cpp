#include "weights.h"
#include <fstream>
#include <iostream>
#include <cstring>

// Helper: escreve um Tensor inteiro no arquivo
static void write_tensor(std::ofstream& f, const Tensor& t) {
    f.write(reinterpret_cast<const char*>(t.data.data()), 
            t.data.size() * sizeof(float));
}

// Helper: lê um Tensor inteiro do arquivo
static void read_tensor(std::ifstream& f, Tensor& t) {
    f.read(reinterpret_cast<char*>(t.data.data()), 
           t.data.size() * sizeof(float));
}

bool save_model(const Model& model, const std::string& filepath) {
    std::ofstream f(filepath, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Erro: nao foi possivel criar " << filepath << std::endl;
        return false;
    }
    
    // 1. Escreve o header
    ModelHeader header = {};
    header.magic = MLLM_MAGIC;
    header.version = MLLM_VERSION;
    header.vocab_size = model.vocab_size;
    header.dim = model.dim;
    header.hidden_dim = model.hidden_dim;
    header.n_layers = model.n_layers;
    header.n_heads = model.n_heads;
    header.max_seq_len = model.max_seq_len;
    
    f.write(reinterpret_cast<const char*>(&header), sizeof(header));
    
    // 2. Escreve token_embedding
    write_tensor(f, model.token_embedding);
    
    // 3. Escreve pesos de cada camada
    for (int l = 0; l < model.n_layers; l++) {
        const auto& layer = model.layers[l];
        write_tensor(f, layer.norm_attn);
        write_tensor(f, layer.wq);
        write_tensor(f, layer.wk);
        write_tensor(f, layer.wv);
        write_tensor(f, layer.wo);
        write_tensor(f, layer.norm_ffn);
        write_tensor(f, layer.w1);
        write_tensor(f, layer.w3);
        write_tensor(f, layer.w2);
    }
    
    // 4. Escreve norm_final e lm_head
    write_tensor(f, model.norm_final);
    write_tensor(f, model.lm_head);
    
    f.close();
    
    // Tamanho do arquivo
    std::ifstream check(filepath, std::ios::binary | std::ios::ate);
    auto size = check.tellg();
    
    std::cout << "✅ Modelo salvo em: " << filepath << std::endl;
    std::cout << "   Tamanho: " << size << " bytes (" 
              << (size / 1024.0f / 1024.0f) << " MB)" << std::endl;
    
    return true;
}

Model* load_model(const std::string& filepath) {
    std::ifstream f(filepath, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Erro: nao foi possivel abrir " << filepath << std::endl;
        return nullptr;
    }
    
    // 1. Lê o header
    ModelHeader header;
    f.read(reinterpret_cast<char*>(&header), sizeof(header));
    
    // 2. Valida magic number
    if (header.magic != MLLM_MAGIC) {
        std::cerr << "Erro: arquivo nao e um modelo MLLM valido!" << std::endl;
        std::cerr << "  Magic esperado: 0x" << std::hex << MLLM_MAGIC << std::endl;
        std::cerr << "  Magic lido:     0x" << std::hex << header.magic << std::dec << std::endl;
        return nullptr;
    }
    
    if (header.version != MLLM_VERSION) {
        std::cerr << "Erro: versao incompativel (" << header.version 
                  << " vs " << MLLM_VERSION << ")" << std::endl;
        return nullptr;
    }
    
    std::cout << "📂 Carregando modelo de: " << filepath << std::endl;
    std::cout << "   vocab=" << header.vocab_size 
              << " dim=" << header.dim 
              << " hidden=" << header.hidden_dim
              << " layers=" << header.n_layers
              << " heads=" << header.n_heads << std::endl;
    
    // 3. Cria o modelo com a config do header
    Model* model = new Model(
        header.vocab_size,
        header.dim,
        header.hidden_dim,
        header.n_layers,
        header.n_heads,
        header.max_seq_len
    );
    
    // 4. Lê token_embedding
    read_tensor(f, model->token_embedding);
    
    // 5. Lê pesos de cada camada
    for (int l = 0; l < model->n_layers; l++) {
        auto& layer = model->layers[l];
        read_tensor(f, layer.norm_attn);
        read_tensor(f, layer.wq);
        read_tensor(f, layer.wk);
        read_tensor(f, layer.wv);
        read_tensor(f, layer.wo);
        read_tensor(f, layer.norm_ffn);
        read_tensor(f, layer.w1);
        read_tensor(f, layer.w3);
        read_tensor(f, layer.w2);
    }
    
    // 6. Lê norm_final e lm_head
    read_tensor(f, model->norm_final);
    read_tensor(f, model->lm_head);
    
    f.close();
    
    std::cout << "✅ Modelo carregado com sucesso!" << std::endl;
    return model;
}