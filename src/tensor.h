#pragma once
#include <vector>
#include <iostream>

// Nossa estrutura de dados básica para guardar pesos e ativações
struct Tensor {
    std::vector<float> data;
    int rows;
    int cols;

    Tensor() : rows(0), cols(0) {}
    Tensor(int r, int c) : data(r * c, 0.0f), rows(r), cols(c) {}

    // Função para acessar como se fosse uma matriz 2D
    float& at(int r, int c) { return data[r * cols + c]; }
    const float& at(int r, int c) const { return data[r * cols + c]; }
};