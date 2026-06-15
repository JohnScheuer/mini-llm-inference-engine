#include <iostream>
#include <vector>
#include <cstdio>

int main() {
    // Dimensões do Stories15M
    int h[7] = {288, 768, 6, 6, 6, 32000, 256};
    FILE* f = fopen("model.bin", "wb");
    fwrite(h, sizeof(int), 7, f);

    // Gera lixo aleatório para simular os pesos (aprox 60MB)
    float dummy = 0.1f;
    for (int i = 0; i < 15000000; i++) {
        fwrite(&dummy, sizeof(float), 1, f);
    }
    fclose(f);
    std::cout << "Arquivo model.bin dummy gerado com sucesso! (60MB)" << std::endl;
    return 0;
}