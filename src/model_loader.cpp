src/model_loader.cpp:
#include "model/model_weights.h"
#include "runtime/model_config.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <cstdio>
#include <cuda_fp16.h>
#include "json.hpp"

using json = nlohmann::json;

namespace runtime {

////////////////////////////////////////////////////////////////
// ====================== BINARY LOADER ========================
////////////////////////////////////////////////////////////////

bool load_binary_model(const std::string& path,
ModelConfig& config,
ModelWeights& weights)
{
FILE* f = fopen(path.c_str(), "rb");
if (!f) {
std::cerr << "Erro ao abrir " << path << std::endl;
return false;
}

text

int h[7];
if (fread(h, sizeof(int), 7, f) != 7) {
    fclose(f);
    return false;
}

config.dim         = h[0];
config.hidden_dim  = h[1];
config.n_layers    = h[2];
config.n_heads     = h[3];
config.vocab_size  = h[5];
config.max_seq_len = h[6];

if (!config.is_valid()) {
    fclose(f);
    return false;
}

weights.dim         = config.dim;
weights.hidden_dim  = config.hidden_dim;
weights.n_layers    = config.n_layers;
weights.vocab_size  = config.vocab_size;

weights.token_embedding.resize(
    (size_t)config.vocab_size * config.dim);

weights.layers.resize(config.n_layers);

auto read_f = [&](float* p, size_t n) -> bool {
    return fread(p, sizeof(float), n, f) == n;
};

read_f(weights.token_embedding.data(),
       weights.token_embedding.size());

for (int i = 0; i < config.n_layers; i++) {

    auto& layer = weights.layers[i];

    layer.norm_attn.resize(config.dim);
    layer.wq.resize((size_t)config.dim * config.dim);
    layer.wk.resize((size_t)config.dim * config.dim);
    layer.wv.resize((size_t)config.dim * config.dim);
    layer.wo.resize((size_t)config.dim * config.dim);

    layer.norm_ffn.resize(config.dim);
    layer.w1.resize((size_t)config.dim * config.hidden_dim);
    layer.w2.resize((size_t)config.hidden_dim * config.dim);
    layer.w3.resize((size_t)config.dim * config.hidden_dim);

    read_f(layer.norm_attn.data(), config.dim);
    read_f(layer.wq.data(), layer.wq.size());
    read_f(layer.wk.data(), layer.wk.size());
    read_f(layer.wv.data(), layer.wv.size());
    read_f(layer.wo.data(), layer.wo.size());
    read_f(layer.norm_ffn.data(), config.dim);
    read_f(layer.w1.data(), layer.w1.size());
    read_f(layer.w2.data(), layer.w2.size());
    read_f(layer.w3.data(), layer.w3.size());
}

weights.norm_final.resize(config.dim);
weights.lm_head.resize(
    (size_t)config.vocab_size * config.dim);

read_f(weights.norm_final.data(), config.dim);
read_f(weights.lm_head.data(), weights.lm_head.size());

fclose(f);
return true;
}

////////////////////////////////////////////////////////////////
// ====================== SAFETENSORS LOADER ===================
////////////////////////////////////////////////////////////////

bool load_safetensors(const std::string& path,
ModelConfig& config,
ModelWeights& weights)
{
std::ifstream f(path, std::ios::binary);
if (!f) {
std::cerr << "Erro ao abrir safetensors\n";
return false;
}

text

uint64_t header_size;
f.read(reinterpret_cast<char*>(&header_size), 8);

std::string header_json(header_size, ' ');
f.read(&header_json[0], header_size);

json meta = json::parse(header_json);
uint64_t data_offset = 8 + header_size;

auto read_tensor = [&](const std::string& name) {

    if (!meta.contains(name))
        throw std::runtime_error("Tensor não encontrado: " + name);

    uint64_t start = meta[name]["data_offsets"][0];
    uint64_t end   = meta[name]["data_offsets"][1];

    f.seekg(data_offset + start);

    size_t bytes = end - start;
    std::vector<uint16_t> h(bytes / 2);
    f.read(reinterpret_cast<char*>(h.data()), bytes);

    std::vector<float> out(h.size());
    std::string dtype = meta[name]["dtype"];

    for (size_t i = 0; i < h.size(); i++) {

        if (dtype == "BF16") {
            uint32_t tmp = ((uint32_t)h[i]) << 16;
            float val;
            std::memcpy(&val, &tmp, sizeof(float));
            out[i] = val;
        }
        else if (dtype == "F16") {
            __half hv;
            std::memcpy(&hv, &h[i], sizeof(uint16_t));
            out[i] = __half2float(hv);
        }
        else {
            throw std::runtime_error("Unsupported dtype: " + dtype);
        }
    }

    return out;
};

// -------- Inferir dimensões --------

if (!meta.contains("model.embed_tokens.weight"))
    return false;

auto embed_shape =
    meta["model.embed_tokens.weight"]["shape"];

config.vocab_size = embed_shape[0];
config.dim        = embed_shape[1];

// Inferir número de layers
int layer_count = 0;
while (meta.contains("model.layers." +
                     std::to_string(layer_count) +
                     ".self_attn.q_proj.weight")) {
    layer_count++;
}

config.n_layers = layer_count;

// Inferir hidden_dim
std::string w1_key =
    "model.layers.0.mlp.gate_proj.weight";

if (!meta.contains(w1_key))
    return false;

auto w1_shape = meta[w1_key]["shape"];
config.hidden_dim = w1_shape[0];

// n_heads não é confiavelmente inferível só pelos pesos.
// Deve ser definido previamente ou via config.json.
if (config.n_heads <= 0) {
    std::cerr << "[Loader] Aviso: n_heads não definido. Usando fallback n_heads=1.\n";
    config.n_heads = 1;
}

// max_seq_len também idealmente vem de config.json
if (config.max_seq_len <= 0) {
    config.max_seq_len = 2048;
}

if (!config.is_valid())
    return false;

// -------- Preencher pesos --------

weights.dim         = config.dim;
weights.hidden_dim  = config.hidden_dim;
weights.n_layers    = config.n_layers;
weights.vocab_size  = config.vocab_size;

weights.token_embedding =
    read_tensor("model.embed_tokens.weight");

weights.layers.resize(config.n_layers);

for (int i = 0; i < config.n_layers; i++) {

    auto& layer = weights.layers[i];
    std::string prefix =
        "model.layers." + std::to_string(i);

    layer.norm_attn =
        read_tensor(prefix + ".input_layernorm.weight");
    layer.wq =
        read_tensor(prefix + ".self_attn.q_proj.weight");
    layer.wk =
        read_tensor(prefix + ".self_attn.k_proj.weight");
    layer.wv =
        read_tensor(prefix + ".self_attn.v_proj.weight");
    layer.wo =
        read_tensor(prefix + ".self_attn.o_proj.weight");

    layer.norm_ffn =
        read_tensor(prefix + ".post_attention_layernorm.weight");
    layer.w1 =
        read_tensor(prefix + ".mlp.gate_proj.weight");
    layer.w2 =
        read_tensor(prefix + ".mlp.down_proj.weight");
    layer.w3 =
        read_tensor(prefix + ".mlp.up_proj.weight");
}

weights.norm_final =
    read_tensor("model.norm.weight");

weights.lm_head =
    read_tensor("lm_head.weight");

f.close();
return true;
}

} // namespace runtime

src/model/model.cpp:
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cuda_fp16.h>
#include "json.hpp"
#include "model.h"

using json = nlohmann::json;

bool load_model_weights(Model& model, const std::string& path) {
FILE* f = fopen(path.c_str(), "rb");
if (!f) {
std::cerr << "Erro: não foi possível abrir " << path << std::endl;
return false;
}

text

int h[7];
if (fread(h, sizeof(int), 7, f) != 7) { fclose(f); return false; }

model.dim        = h[0];
model.hidden_dim = h[1];
model.n_layers   = h[2];
model.n_heads    = h[3];
model.vocab_size = h[5];
model.max_seq_len= h[6];

std::cout << "[Model] dim=" << model.dim
          << " hidden=" << model.hidden_dim
          << " layers=" << model.n_layers
          << " heads=" << model.n_heads
          << " vocab=" << model.vocab_size
          << " max_seq=" << model.max_seq_len << std::endl;

model.token_embedding = Tensor(model.vocab_size, model.dim);
model.layers.clear();
for (int i = 0; i < model.n_layers; i++)
    model.layers.emplace_back(model.dim, model.hidden_dim);
model.norm_final = Tensor(1, model.dim);
model.lm_head    = Tensor(model.vocab_size, model.dim);

auto read_f = [&](float* p, size_t n) -> bool {
    return fread(p, sizeof(float), n, f) == n;
};

read_f(model.token_embedding.data.data(),
       (size_t)model.vocab_size * model.dim);

for (int i = 0; i < model.n_layers; i++)
    read_f(model.layers[i].norm_attn.data.data(), model.dim);
for (int i = 0; i < model.n_layers; i++)
    read_f(model.layers[i].wq.data.data(),
           (size_t)model.dim * model.dim);
for (int i = 0; i < model.n_layers; i++)
    read_f(model.layers[i].wk.data.data(),
           (size_t)model.dim * model.dim);
for (int i = 0; i < model.n_layers; i++)
    read_f(model.layers[i].wv.data.data(),
           (size_t)model.dim * model.dim);
for (int i = 0; i < model.n_layers; i++)
    read_f(model.layers[i].wo.data.data(),
           (size_t)model.dim * model.dim);
for (int i = 0; i < model.n_layers; i++)
    read_f(model.layers[i].norm_ffn.data.data(), model.dim);
for (int i = 0; i < model.n_layers; i++)
    read_f(model.layers[i].w1.data.data(),
           (size_t)model.dim * model.hidden_dim);
for (int i = 0; i < model.n_layers; i++)
    read_f(model.layers[i].w2.data.data(),
           (size_t)model.hidden_dim * model.dim);
for (int i = 0; i < model.n_layers; i++)
    read_f(model.layers[i].w3.data.data(),
           (size_t)model.dim * model.hidden_dim);

read_f(model.norm_final.data.data(), model.dim);
read_f(model.lm_head.data.data(),
       (size_t)model.vocab_size * model.dim);

fclose(f);
std::cout << "[Model] Pesos carregados com sucesso!" << std::endl;
return true;
}
void load_safetensors_improved(const std::string& path, Model& m)
{
std::ifstream f(path, std::ios::binary);
if (!f) {
std::cerr << "Erro ao abrir Safetensors!" << std::endl;
exit(1);
}

text

uint64_t header_size;
f.read(reinterpret_cast<char*>(&header_size), 8);

std::string header_json(header_size, ' ');
f.read(&header_json[0], header_size);

json meta = json::parse(header_json);
uint64_t data_offset = 8 + header_size;

auto read_tensor = [&](const std::string& name) {
    uint64_t start = meta[name]["data_offsets"][0];
    uint64_t end   = meta[name]["data_offsets"][1];

    f.seekg(data_offset + start);

    size_t bytes = end - start;
    std::vector<uint16_t> h(bytes / 2);
    f.read(reinterpret_cast<char*>(h.data()), bytes);

    std::vector<float> out(h.size());

    std::string dtype = meta[name]["dtype"];

    for (size_t i = 0; i < h.size(); i++)
    {
        if (dtype == "BF16")
        {
            uint32_t tmp = ((uint32_t)h[i]) << 16;
            float val;
            std::memcpy(&val, &tmp, sizeof(float));
            out[i] = val;
        }
        else if (dtype == "F16")
        {
            __half hv;
            std::memcpy(&hv, &h[i], sizeof(uint16_t));
            out[i] = __half2float(hv);
        }
        else
        {
            std::cerr << "Unsupported dtype: " << dtype << std::endl;
            exit(1);
        }
    }

    return out;
};

m.token_embedding.data = read_tensor("model.embed_tokens.weight");

for (int i = 0; i < m.n_layers; i++)
{
    std::string prefix = "model.layers." + std::to_string(i);

    m.layers[i].norm_attn.data = read_tensor(prefix + ".input_layernorm.weight");
    m.layers[i].wq.data = read_tensor(prefix + ".self_attn.q_proj.weight");
    m.layers[i].wk.data = read_tensor(prefix + ".self_attn.k_proj.weight");
    m.layers[i].wv.data = read_tensor(prefix + ".self_attn.v_proj.weight");
    m.layers[i].wo.data = read_tensor(prefix + ".self_attn.o_proj.weight");

    m.layers[i].norm_ffn.data = read_tensor(prefix + ".post_attention_layernorm.weight");
    m.layers[i].w1.data = read_tensor(prefix + ".mlp.gate_proj.weight");
    m.layers[i].w2.data = read_tensor(prefix + ".mlp.down_proj.weight");
    m.layers[i].w3.data = read_tensor(prefix + ".mlp.up_proj.weight");
}

m.norm_final.data = read_tensor("model.norm.weight");
m.lm_head.data = read_tensor("lm_head.weight");

std::cout << "[Loader] ✓ Modelo carregado com sucesso!" << std::endl;
}