#pragma once

#include "tensor.h"
#include <fstream>
#include <vector>

// -------------------------------------------------------------------------
// DECLARAÇÕES DE I/O DE DISCO (READ / WRITE)
// -------------------------------------------------------------------------

// Leitura e Escrita em formato FP32 (Padrão do seu model.bin atual, 4 bytes)
void write_tensor_fp32(std::ofstream& out, const Tensor& t);
void read_tensor_fp32(std::ifstream& in, Tensor& t);

// Leitura e Escrita em formato FP16 (Comprimido, 2 bytes)
void write_tensor_fp16(std::ofstream& out, const Tensor& t);
void read_tensor_fp16(std::ifstream& in, Tensor& t);


// -------------------------------------------------------------------------
// DECLARAÇÕES DE CONVERSÃO DE HARDWARE
// -------------------------------------------------------------------------

// Expostas caso você precise converter vetores avulsos em outras partes do código
void tensor_to_fp16(const Tensor& t, std::vector<unsigned short>& out_fp16);
void fp16_to_tensor(const std::vector<unsigned short>& in_fp16, Tensor& t);