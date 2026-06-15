#include <fstream>
#include <vector>
#include <random>
#include <iostream>

int main() {

    // Stories15M config
    int dim = 288;
    int hidden_dim = 768;
    int n_layers = 6;
    int n_heads = 6;
    int n_kv_heads = 6;
    int vocab_size = 32000;
    int max_seq_len = 256;

    std::ofstream out("model.bin", std::ios::binary);

    // Header
    int header[7] = {
        dim,
        hidden_dim,
        n_layers,
        n_heads,
        n_kv_heads,
        vocab_size,
        max_seq_len
    };

    out.write(reinterpret_cast<char*>(header), sizeof(header));

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.02f);

    auto write_random = [&](size_t count) {
        std::vector<float> buffer(count);
        for (size_t i = 0; i < count; i++)
            buffer[i] = dist(rng);
        out.write(reinterpret_cast<char*>(buffer.data()),
                  count * sizeof(float));
    };

    auto write_ones = [&](size_t count) {
        std::vector<float> buffer(count, 1.0f);
        out.write(reinterpret_cast<char*>(buffer.data()),
                  count * sizeof(float));
    };

    // 1. Token Embedding
    write_random((size_t)vocab_size * dim);

    // 2. RMS Attn
    for (int i = 0; i < n_layers; i++)
        write_ones(dim);

    // 3. WQ, WK, WV, WO
    for (int i = 0; i < n_layers; i++)
        write_random((size_t)dim * dim);
    for (int i = 0; i < n_layers; i++)
        write_random((size_t)dim * dim);
    for (int i = 0; i < n_layers; i++)
        write_random((size_t)dim * dim);
    for (int i = 0; i < n_layers; i++)
        write_random((size_t)dim * dim);

    // 4. RMS FFN
    for (int i = 0; i < n_layers; i++)
        write_ones(dim);

    // 5. W1, W2, W3
    for (int i = 0; i < n_layers; i++)
        write_random((size_t)dim * hidden_dim);
    for (int i = 0; i < n_layers; i++)
        write_random((size_t)hidden_dim * dim);
    for (int i = 0; i < n_layers; i++)
        write_random((size_t)dim * hidden_dim);

    // 6. Final Norm
    write_ones(dim);

    // 7. LM Head
    write_random((size_t)vocab_size * dim);

    out.close();

    std::cout << "✅ model.bin gerado com sucesso.\n";
}