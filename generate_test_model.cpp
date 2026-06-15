#include <fstream>
#include <vector>
#include <iostream>

int main() {
    int dim = 288; int hidden_dim = 768; int n_layers = 6;
    int n_heads = 6; int n_kv_heads = 6; int vocab_size = 32000; int max_seq_len = 256;

    std::ofstream out("model.bin", std::ios::binary);
    int header[7] = {dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len};
    out.write((char*)header, sizeof(header));

    auto write_zeros = [&](size_t count) {
        std::vector<float> b(count, 0.0f);
        out.write((char*)b.data(), count * sizeof(float));
    };

    // 1. Token Embedding: Colocamos o valor 10.0f na posição 'id % dim'
    // para cada token. Isso cria uma "assinatura" única.
    for (int i = 0; i < vocab_size; i++) {
        std::vector<float> row(dim, 0.0f);
        row[i % dim] = 10.0f; 
        out.write((char*)row.data(), dim * sizeof(float));
    }

    // 2. RMS Norms (Attn e FFN) e Tabelas: Tudo 1.0
    for (int i = 0; i < n_layers * 2 + 1; i++) {
        std::vector<float> b(dim, 1.0f);
        out.write((char*)b.data(), dim * sizeof(float));
    }

    // 3. Matrizes Internas (WQ, WK, WV, WO, W1, W2, W3): Identidade simples ou Zeros
    // Para teste de fluxo, vamos apenas zerar o corpo do transformer (pass-through)
    size_t body_size = (size_t)n_layers * (4 * dim * dim + 3 * dim * hidden_dim);
    write_zeros(body_size);

    // 4. LM Head: O segredo do teste.
    // Colocamos o valor 10.0f na mesma posição 'id % dim'.
    // Se o motor INT8 multiplicar corretamente, o token de entrada X
    // vai dar o maior score no token de saída X.
    for (int i = 0; i < vocab_size; i++) {
        std::vector<float> row(dim, 0.0f);
        row[i % dim] = 10.0f; 
        out.write((char*)row.data(), dim * sizeof(float));
    }

    out.close();
    std::cout << "✅ Modelo de Identidade Gerado.\n";
    return 0;
}