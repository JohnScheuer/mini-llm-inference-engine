#include "transformer.h"
#include <omp.h>

void transformer_block_forward(
    Tensor& out, const Tensor& x,
    TransformerLayer& layer,
    int pos, int dim, int hidden_dim, int n_heads
) {
    // SUB-BLOCO 1: Attention
    
    Tensor normed(1, dim);
    rmsnorm(normed, x, layer.norm_attn);
    
    Tensor attn_out(1, dim);
    attention_forward(attn_out, normed,
                      layer.wq, layer.wk, layer.wv, layer.wo,
                      layer.cache, pos, dim, n_heads);
    
    Tensor h(1, dim);
    #pragma omp parallel for
    for (int i = 0; i < dim; i++) {
        h.data[i] = x.data[i] + attn_out.data[i];
    }
    
    // SUB-BLOCO 2: FFN
    
    rmsnorm(normed, h, layer.norm_ffn);
    
    Tensor ffn_out(1, dim);
    ffn_forward(ffn_out, normed,
                layer.w1, layer.w2, layer.w3,
                dim, hidden_dim);
    
    #pragma omp parallel for
    for (int i = 0; i < dim; i++) {
        out.data[i] = h.data[i] + ffn_out.data[i];
    }
}