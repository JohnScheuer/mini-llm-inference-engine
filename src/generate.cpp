#include "generate.h"
#include "layers.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>

// Gerador de números aleatórios (compartilhado)
static std::mt19937 rng(42);  // seed fixa para reprodutibilidade

// 1. ARGMAX: pega o token com maior logit (determinístico)
int sample_argmax(const Tensor& logits) {
    int best = 0;
    float best_val = logits.data[0];
    for (int i = 1; i < logits.cols; i++) {
        if (logits.data[i] > best_val) {
            best_val = logits.data[i];
            best = i;
        }
    }
    return best;
}

// 2. TEMPERATURE: divide logits por temperature, aplica softmax, sampleia
int sample_temperature(const Tensor& logits, float temperature) {
    int vocab_size = logits.cols;
    std::vector<float> probs(vocab_size);
    
    // Copia e divide por temperature
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = logits.data[i] / temperature;
    }
    
    // Softmax
    softmax(probs.data(), vocab_size);
    
    // Sampleia da distribuição
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(rng);
}

// 3. TOP-K: pega apenas os K mais prováveis e sampleia entre eles
int sample_top_k(const Tensor& logits, float temperature, int top_k) {
    int vocab_size = logits.cols;
    
    // Cria par (logit, índice)
    std::vector<std::pair<float, int>> indexed_logits(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        indexed_logits[i] = {logits.data[i] / temperature, i};
    }
    
    // Pega os top_k maiores
    std::partial_sort(
        indexed_logits.begin(), 
        indexed_logits.begin() + top_k, 
        indexed_logits.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );
    
    // Softmax APENAS nos top_k
    std::vector<float> top_probs(top_k);
    for (int i = 0; i < top_k; i++) {
        top_probs[i] = indexed_logits[i].first;
    }
    softmax(top_probs.data(), top_k);
    
    // Sampleia entre os top_k
    std::discrete_distribution<int> dist(top_probs.begin(), top_probs.end());
    int idx = dist(rng);
    return indexed_logits[idx].second;
}

// FUNÇÃO PRINCIPAL: geração de tokens
std::vector<int> generate(
    Model& model,
    const std::vector<int>& prompt,
    int max_new_tokens,
    float temperature,
    int top_k
) {
    std::vector<int> tokens = prompt;
    Tensor logits(1, model.vocab_size);
    
    // 1. PREFILL: processa o prompt todo para preencher o KV-Cache
    std::cout << "Processando prompt (" << prompt.size() << " tokens)..." << std::endl;
    for (size_t i = 0; i < prompt.size(); i++) {
        model_forward(logits, model, prompt[i], i);
    }
    
    // 2. GERAÇÃO: loop autoregressivo
    std::cout << "Gerando " << max_new_tokens << " tokens..." << std::endl;
    int pos = prompt.size();
    
    for (int step = 0; step < max_new_tokens; step++) {
        // Sampleia o próximo token a partir dos últimos logits
        int next_token;
        if (temperature <= 0.0f) {
            next_token = sample_argmax(logits);
        } else if (top_k > 0) {
            next_token = sample_top_k(logits, temperature, top_k);
        } else {
            next_token = sample_temperature(logits, temperature);
        }
        
        tokens.push_back(next_token);
        std::cout << next_token << " ";
        std::cout.flush();
        
        // Verifica se atingiu o limite
        if (pos >= model.max_seq_len - 1) break;
        
        // Forward com o token recém-gerado
        model_forward(logits, model, next_token, pos);
        pos++;
    }
    std::cout << std::endl;
    
    return tokens;
}