#include "weights.h"
#include "quantize.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <vector>

// Escreve um Tensor em FP32
static void write_tensor_fp32(std::ofstream& f, const Tensor& t) {
    f.write(reinterpret_cast<const char*>(t.data.data()), 
            t.data.size() * sizeof(float));
}

// Escreve um Tensor em FP16 (convertendo de FP32)
static void write_tensor_fp16(std::ofstream& f, const Tensor& t) {
    std::vector<uint16_t> fp16_data;
    tensor_to_fp16(t, fp16_data);
    f.write(reinterpret_cast<const char*>(fp16_data.data()), 
            fp16_data.size() * sizeof(uint16_t));
}

// Lê um Tensor em FP32
static void read_tensor_fp32(std::ifstream& f, Tensor& t) {
    f.read(reinterpret_cast<char*>(t.data.data()), 
           t.data.size() * sizeof(float));
}

// Lê um Tensor em FP16 (convertendo para FP32 na RAM)
static void read_tensor_fp16(std::ifstream& f, Tensor& t) {
    std::vector<uint16_t> fp16_data(t.data.size());
    f.read(reinterpret_cast<char*>(fp16_data.data()), 
           fp16_data.size() * sizeof(uint16_t));
    fp16_to_tensor(fp16_data, t);
}

bool save_model(const Model& model, const std::string& filepath, QuantType quant) {
    std::ofstream f(filepath, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Erro: nao foi possivel criar " << filepath << std::endl;
        return false;
    }
    
    // Escreve header
    ModelHeader header = {};
    header.magic = MLLM_MAGIC;
    header.version = MLLM_VERSION;
    header.quant_type = quant;
    header.vocab_size = model.vocab_size;
    header.dim = model.dim;
    header.hidden_dim = model.hidden_dim;
    header.n_layers = model.n_layers;
    header.n_heads = model.n_heads;
    header.max_seq_len = model.max_seq_len;
    
    f.write(reinterpret_cast<const char*>(&header), sizeof(header));
    
    // Escolhe a função de escrita baseada na quantização
    auto write_fn = (quant == QUANT_FP16) ? write_tensor_fp16 : write_tensor_fp32;
    
    write_fn(f, model.token_embedding);
    
    for (int l = 0; l < model.n_layers; l++) {
        const auto& layer = model.layers[l];
        write_fn(f, layer.norm_attn);
        write_fn(f, layer.wq);
        write_fn(f, layer.wk);
        write_fn(f, layer.wv);
        write_fn(f, layer.wo);
        write_fn(f, layer.norm_ffn);
        write_fn(f, layer.w1);
        write_fn(f, layer.w3);
        write_fn(f, layer.w2);
    }
    
    write_fn(f, model.norm_final);
    write_fn(f, model.lm_head);
    
    f.close();
    
    std::ifstream check(filepath, std::ios::binary | std::ios::ate);
    auto size = check.tellg();
    
    std::cout << "Modelo salvo em: " << filepath << std::endl;
    std::cout << "   Quantizacao: " << (quant == QUANT_FP16 ? "FP16" : "FP32") << std::endl;
    std::cout << "   Tamanho: " << size << " bytes (" 
              << (size / 1024.0f) << " KB)" << std::endl;
    
    return true;
}

Model* load_model(const std::string& filepath) {
    std::ifstream f(filepath, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Erro: nao foi possivel abrir " << filepath << std::endl;
        return nullptr;
    }
    
    ModelHeader header;
    f.read(reinterpret_cast<char*>(&header), sizeof(header));
    
    if (header.magic != MLLM_MAGIC) {
        std::cerr << "Erro: arquivo nao e um modelo MLLM valido!" << std::endl;
        return nullptr;
    }
    
    if (header.version != MLLM_VERSION) {
        std::cerr << "Erro: versao incompativel" << std::endl;
        return nullptr;
    }
    
    QuantType quant = static_cast<QuantType>(header.quant_type);
    
    std::cout << "Carregando modelo de: " << filepath << std::endl;
    std::cout << "   Quantizacao: " << (quant == QUANT_FP16 ? "FP16" : "FP32") << std::endl;
    std::cout << "   vocab=" << header.vocab_size 
              << " dim=" << header.dim 
              << " layers=" << header.n_layers << std::endl;
    
    Model* model = new Model(
        header.vocab_size,
        header.dim,
        header.hidden_dim,
        header.n_layers,
        header.n_heads,
        header.max_seq_len
    );
    
    // Escolhe função de leitura baseada na quantização
    auto read_fn = (quant == QUANT_FP16) ? read_tensor_fp16 : read_tensor_fp32;
    
    read_fn(f, model->token_embedding);
    
    for (int l = 0; l < model->n_layers; l++) {
        auto& layer = model->layers[l];
        read_fn(f, layer.norm_attn);
        read_fn(f, layer.wq);
        read_fn(f, layer.wk);
        read_fn(f, layer.wv);
        read_fn(f, layer.wo);
        read_fn(f, layer.norm_ffn);
        read_fn(f, layer.w1);
        read_fn(f, layer.w3);
        read_fn(f, layer.w2);
    }
    
    read_fn(f, model->norm_final);
    read_fn(f, model->lm_head);
    
    f.close();
    
    std::cout << "Modelo carregado!" << std::endl;
    return model;
}