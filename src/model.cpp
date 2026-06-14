#include "model.h"
#include "layers.h"
#include "transformer.h"
#include "matmul_blocked.h"
#include <cstdio>
#include <cstring>
#include <iostream>

void model_forward(Tensor& logits, Model& model, int token_id, int pos) {
    int dim = model.dim;
    Tensor x(1, dim);
    
    // 1. Embedding lookup: Puxa o vetor do token atual
    float* embed_row = &model.token_embedding.data[token_id * dim];
    std::memcpy(x.data.data(), embed_row, dim * sizeof(float));

    // 2. Transformer Layers: Loop pelas N camadas
    Tensor x_next(1, dim);
    for (int l = 0; l < model.n_layers; l++) {
        transformer_block_forward(x_next, x, model.layers[l], pos, dim, model.hidden_dim, model.n_heads);
        // Residual connection swap
        std::memcpy(x.data.data(), x_next.data.data(), dim * sizeof(float));
    }

    // 3. Final RMSNorm
    Tensor normed(1, dim);
    rmsnorm(normed, x, model.norm_final);

    // 4. LM HEAD (Logits): Projeção final otimizada para formato Llama [Vocab x Dim]
    // Como M=1 na geração, processamos como uma série de Dot Products
    float* l_ptr = logits.data.data();
    float* w_ptr = model.lm_head.data.data();
    float* n_ptr = normed.data.data();

    #pragma omp parallel for schedule(static)
    for (int v = 0; v < model.vocab_size; v++) {
        float score = 0;
        float* weight_row = &w_ptr[v * dim];
        for (int d = 0; d < dim; d++) {
            score += n_ptr[d] * weight_row[d];
        }
        l_ptr[v] = score;
    }
}

bool load_model_weights(Model& model, const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;

    // Lendo o cabeçalho (7 ints)
    int h[7];
    if (fread(h, sizeof(int), 7, f) != 7) { fclose(f); return false; }
    
    // Configurações do modelo
    bool shared = h[5] < 0; // vocab_size negativo indica pesos compartilhados
    model.dim = h[0];
    model.hidden_dim = h[1];
    model.n_layers = h[2];
    model.n_heads = h[3];
    model.vocab_size = shared ? -h[5] : h[5];
    model.max_seq_len = h[6];

    // Alocação física dos tensores
    model.token_embedding = Tensor(model.vocab_size, model.dim);
    model.layers.clear();
    for(int i=0; i<model.n_layers; i++) {
        model.layers.emplace_back(model.dim, model.hidden_dim, model.max_seq_len);
    }
    model.norm_final = Tensor(1, model.dim);
    model.lm_head = Tensor(model.vocab_size, model.dim);

    auto read_f = [&](float* p, size_t n) { return fread(p, sizeof(float), n, f) == n; };

    // --- ORDEM DE LEITURA BINÁRIA (Padrão llama2.c) ---
    
    // 1. Embedding Table
    read_f(model.token_embedding.data.data(), (size_t)model.vocab_size * model.dim);

    // 2. RMS Attn weights
    for (int i = 0; i < model.n_layers; i++) read_f(model.layers[i].norm_attn.data.data(), model.dim);
    
    // 3. WQ, WK, WV, WO
    for (int i = 0; i < model.n_layers; i++) read_f(model.layers[i].wq.data.data(), (size_t)model.dim * model.dim);
    for (int i = 0; i < model.n_layers; i++) read_f(model.layers[i].wk.data.data(), (size_t)model.dim * model.dim);
    for (int i = 0; i < model.n_layers; i++) read_f(model.layers[i].wv.data.data(), (size_t)model.dim * model.dim);
    for (int i = 0; i < model.n_layers; i++) read_f(model.layers[i].wo.data.data(), (size_t)model.dim * model.dim);
    
    // 4. RMS FFN weights
    for (int i = 0; i < model.n_layers; i++) read_f(model.layers[i].norm_ffn.data.data(), model.dim);
    
    // 5. W1, W2, W3
    for (int i = 0; i < model.n_layers; i++) read_f(model.layers[i].w1.data.data(), (size_t)model.dim * model.hidden_dim);
    for (int i = 0; i < model.n_layers; i++) read_f(model.layers[i].w2.data.data(), (size_t)model.hidden_dim * model.dim);
    for (int i = 0; i < model.n_layers; i++) read_f(model.layers[i].w3.data.data(), (size_t)model.dim * model.hidden_dim);

    // 6. Final Norm
    read_f(model.norm_final.data.data(), model.dim);

    // 7. Pular tabelas RoPE pre-computadas
    int head_dim = model.dim / model.n_heads;
    fseek(f, (size_t)model.max_seq_len * head_dim * sizeof(float), SEEK_CUR);

    // 8. LM Head (Lógica de compartilhamento robusta)
    if (shared) {
        std::memcpy(model.lm_head.data.data(), model.token_embedding.data.data(), (size_t)model.vocab_size * model.dim * sizeof(float));
    } else {
        // Tenta ler do arquivo. Se falhar, faz o fallback para o embedding
        if (!read_f(model.lm_head.data.data(), (size_t)model.vocab_size * model.dim)) {
            std::memcpy(model.lm_head.data.data(), model.token_embedding.data.data(), (size_t)model.vocab_size * model.dim * sizeof(float));
        }
    }

    fclose(f);
    return true;
}