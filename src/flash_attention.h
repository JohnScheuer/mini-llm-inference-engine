#pragma once
constexpr int FLASH_BLOCK = 64;
void flash_attention(float* O, const float* Q, const float* K, const float* V, int seq, int dim);
void benchmark_flash(int seq, int dim);
