#include "model.h"
#include "layers.h"
#include "matmul_blocked.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <immintrin.h>
#include <omp.h>

int argmax(const float* v, int size) {
    int max_i = 0;
    float max_p = v[0];
    for (int i = 1; i < size; i++) {
        if (v[i] > max_p) { max_i = i; max_p = v[i]; }
    }
    return max_i;
}

inline void residual_add(float* a, const float* b, int dim) {
    int i = 0;
    for (; i <= dim - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(a + i, _mm256_add_ps(va, vb));
    }
    for (; i < dim; i++) a[i] += b[i];
}

int main(int argc, char* argv[]) {
    // --- 0. Configurar Threads (cores físicos apenas) ---
    int num_physical_cores = omp_get_max_threads() / 2;
    if (num_physical_cores < 1) num_physical_cores = 1;
    omp_set_num_threads(num_physical_cores);

    // --- 1. Carregar Modelo ---
    std::string model_path = "models/stories15M.bin";
    if (argc > 1) model_path = argv[1];

    Model m;
    if (!load_model_weights(m, model_path)) {
        std::cerr << "Falha ao carregar modelo!" << std::endl;
        return 1;
    }

    // --- 2. Quantizar Pesos ---
    std::cout << "[Init] Quantizando pesos para INT8..." << std::endl;
    quantize_model_weights(m);

    // --- 3. Inicializar RunState ---
    RunState s;
    init_run_state(s, m);

    // --- 4. Inicializar KV Cache ---
    KVCache cache(m.max_seq_len, m.dim);

    // --- 5. Configurar Prompt ---
    std::vector<int> prompt_tokens = {1};
    int max_gen_tokens = 100;

    int token = prompt_tokens[0];
    int next;
    int total_tokens_generated = 0;

    std::cout << "[Init] Iniciando inferência..." << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    // --- 6. Loop de Inferência ---
    for (int pos = 0; pos < m.max_seq_len; pos++) {

        float* emb_row = m.token_embedding.data.data()
                         + token * m.dim;
        std::memcpy(s.x.data(), emb_row, m.dim * sizeof(float));

        for (int l = 0; l < m.n_layers; l++) {
            rmsnorm(s.xb.data(), s.x.data(),
                    m.layers[l].norm_attn.data.data(),
                    m.dim, 1e-5f);

            attention_forward(
                s.attn_out.data(), s.xb.data(),
                m.layers[l].wq, m.layers[l].wk,
                m.layers[l].wv, m.layers[l].wo,
                cache,
                s.q.data(), s.k.data(),
                s.v.data(), s.scores.data(),
                s.rope_cos.data() + pos * m.dim,
                s.rope_sin.data() + pos * m.dim,
                pos, m.dim, m.n_heads,
                m.max_seq_len, s.xq.data());

            residual_add(s.x.data(), s.attn_out.data(), m.dim);

            rmsnorm(s.xb.data(), s.x.data(),
                    m.layers[l].norm_ffn.data.data(),
                    m.dim, 1e-5f);

            ffn_forward(
                s.ffn_out.data(), s.xb.data(),
                m.layers[l].w1, m.layers[l].w2, m.layers[l].w3,
                s.ffn_g.data(), s.ffn_u.data(),
                m.dim, m.hidden_dim, s.xq.data());

            residual_add(s.x.data(), s.ffn_out.data(), m.dim);
        }

        rmsnorm(s.x.data(), s.x.data(),
                m.norm_final.data.data(), m.dim, 1e-5f);

        matmul_blocked_int8(
            1, m.vocab_size, m.dim,
            m.lm_head.q_data.data(),
            s.x.data(), s.logits.data(),
            s.xq.data());

        if (pos < (int)prompt_tokens.size() - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = argmax(s.logits.data(), m.vocab_size);
            total_tokens_generated++;

            if (total_tokens_generated >= max_gen_tokens)
                break;
        }
        token = next;
    }

    // --- 7. Performance ---
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    double tok_s = total_tokens_generated / elapsed;

    std::cout << "\n========================================" << std::endl;
    std::cout << " Mini-LLM Inference Engine (HPC Focus)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << " Model:          Stories15M" << std::endl;
    std::cout << " Quantization:   INT8 (W8A32)" << std::endl;
    std::cout << " Hardware:       " << num_physical_cores
              << " threads (physical cores)" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << " Tokens Gerados: " << total_tokens_generated << std::endl;
    std::cout << " Tempo Total:    " << elapsed << "s" << std::endl;
    std::cout << " Throughput:     " << tok_s << " tok/s" << std::endl;
    std::cout << " Latência:       " << (elapsed / total_tokens_generated) * 1000
              << " ms/tok" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}