#include "weights.h"  
#include "tensor.h"   
#include <fstream>
#include <vector>
#include <immintrin.h> 

// -------------------------------------------------------------------------
// LEITURA NATIVA EM FP32 (Para o seu model.bin atual)
// -------------------------------------------------------------------------

void write_tensor_fp32(std::ofstream& out, const Tensor& t) {
    size_t total_elements = t.data.size();
    out.write(reinterpret_cast<const char*>(t.data.data()), total_elements * sizeof(float));
}

void read_tensor_fp32(std::ifstream& in, Tensor& t) {
    size_t total_elements = t.data.size();
    in.read(reinterpret_cast<char*>(t.data.data()), total_elements * sizeof(float));
}

// -------------------------------------------------------------------------
// CONVERSÕES DE HARDWARE (FLOAT32 <-> FLOAT16)
// -------------------------------------------------------------------------

void tensor_to_fp16(const Tensor& t, std::vector<unsigned short>& out_fp16) {
    size_t total_elements = t.data.size();
    out_fp16.resize(total_elements);
    for (size_t i = 0; i < total_elements; ++i) {
        out_fp16[i] = _cvtss_sh(t.data[i], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }
}

void fp16_to_tensor(const std::vector<unsigned short>& in_fp16, Tensor& t) {
    size_t total_elements = in_fp16.size();
    if (t.data.size() != total_elements) t.data.resize(total_elements);
    for (size_t i = 0; i < total_elements; ++i) {
        t.data[i] = _cvtsh_ss(in_fp16[i]);
    }
}

// -------------------------------------------------------------------------
// LEITURA EM FP16 (Para o futuro)
// -------------------------------------------------------------------------

void write_tensor_fp16(std::ofstream& out, const Tensor& t) {
    std::vector<unsigned short> fp16_data;
    tensor_to_fp16(t, fp16_data);
    out.write(reinterpret_cast<const char*>(fp16_data.data()), fp16_data.size() * sizeof(unsigned short));
}

void read_tensor_fp16(std::ifstream& in, Tensor& t) {
    size_t total_elements = t.data.size();
    std::vector<unsigned short> fp16_data(total_elements);
    in.read(reinterpret_cast<char*>(fp16_data.data()), total_elements * sizeof(unsigned short));
    fp16_to_tensor(fp16_data, t);
}