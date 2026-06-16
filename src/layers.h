#ifndef LAYERS_H
#define LAYERS_H

#include "model.h"

// RMSNorm vetorizado (AVX2)
void rmsnorm(float* o, const float* x, const float* gamma,
             int dim, float eps);

void softmax(float* x, int size);

// RoPE com tabelas pré-computadas
void apply_rope_precomputed(float* data, const float* cos_buf,
                            const float* sin_buf, int dim);

// Dot product AVX2
float dot_product_avx2(const float* a, const float* b, int size);

// Attention com RunState (zero alocações)
void attention_forward(float* attn_out, const float* x,
                       const Tensor& wq, const Tensor& wk,
                       const Tensor& wv, const Tensor& wo,
                       KVCache& cache,
                       float* q_buf, float* k_buf,
                       float* v_buf, float* scores_buf,
                       const float* rope_cos, const float* rope_sin,
                       int pos, int dim, int n_heads,
                       int max_seq_len, BlockQ8_0* xq_buf);

// FFN com RunState (zero alocações)
void ffn_forward(float* out, const float* x,
                 const Tensor& w1, const Tensor& w2, const Tensor& w3,
                 float* ffn_g_buf, float* ffn_u_buf,
                 int dim, int hidden_dim, BlockQ8_0* xq_buf);

// Quantização do modelo
void quantize_tensor_to_int8(Tensor& t);
void quantize_model_weights(Model& m);

#endif