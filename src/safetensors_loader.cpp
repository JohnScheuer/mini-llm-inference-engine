#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include "json.hpp" // A biblioteca que baixamos
#include "model.h"

using json = nlohmann::json;

// Função para converter BFloat16 ou Float16 para Float32 (seu motor usa float32 para carregar)
float half_to_float(uint16_t h) {
    unsigned int sign = (h >> 15) & 0x1;
    unsigned int exp = (h >> 10) & 0x1f;
    unsigned int mant = h & 0x3ff;
    unsigned int f_val;
    if (exp == 0) f_val = (sign << 31);
    else if (exp == 31) f_val = (sign << 31) | (0xff << 23) | (mant << 13);
    else f_val = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    return *reinterpret_cast<float*>(&f_val);
}

void load_safetensors(const std::string& path, Model& m) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Erro: Não foi possível abrir " << path << std::endl; exit(1); }

    // 1. Ler os primeiros 8 bytes (Tamanho do JSON)
    uint64_t header_size;
    f.read(reinterpret_cast<char*>(&header_size), 8);

    // 2. Ler e parsear o JSON
    std::vector<char> header_buf(header_size);
    f.read(header_buf.data(), header_size);
    json metadata = json::parse(header_buf.begin(), header_buf.end());

    uint64_t data_start = 8 + header_size;

    // 3. Lambda para carregar tensores específicos
    auto load_t = [&](const std::string& name, std::vector<float>& target) {
        if (metadata.find(name) == metadata.end()) {
            std::cout << "[Warning] Tensor " << name << " não encontrado!" << std::endl;
            return;
        }
        auto info = metadata[name];
        uint64_t start = info["data_offsets"][0];
        uint64_t end = info["data_offsets"][1];
        size_t num_elements = (end - start) / 2; // Safetensors do TinyLlama são 16-bit

        std::vector<uint16_t> raw_data(num_elements);
        f.seekg(data_start + start);
        f.read(reinterpret_cast<char*>(raw_data.data()), end - start);

        target.resize(num_elements);
        for(size_t i=0; i<num_elements; i++) target[i] = half_to_float(raw_data[i]);
    };

    // 4. Mapeamento Llama (TinyLlama)
    std::cout << "[Safetensors] Mapeando tensores para arquitetura Llama..." << std::endl;
    load_t("model.embed_tokens.weight", m.token_embedding.data);
    
    for(int i=0; i<m.n_layers; i++) {
        std::string p = "model.layers." + std::to_string(i);
        load_t(p + ".input_layernorm.weight", m.layers[i].norm_attn.data);
        load_t(p + ".self_attn.q_proj.weight", m.layers[i].wq.data);
        load_t(p + ".self_attn.k_proj.weight", m.layers[i].wk.data);
        load_t(p + ".self_attn.v_proj.weight", m.layers[i].wv.data);
        load_t(p + ".self_attn.o_proj.weight", m.layers[i].wo.data);
        load_t(p + ".post_attention_layernorm.weight", m.layers[i].norm_ffn.data);
        load_t(p + ".mlp.gate_proj.weight", m.layers[i].w1.data);
        load_t(p + ".mlp.down_proj.weight", m.layers[i].w2.data);
        load_t(p + ".mlp.up_proj.weight", m.layers[i].w3.data);
    }
    load_t("model.norm.weight", m.norm_final.data);
    load_t("lm_head.weight", m.lm_head.data);
}