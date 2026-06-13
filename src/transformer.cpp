#include "transformer.h"
#include "layers.h"
#include <omp.h>

void transformer_block_forward(
    Tensor& out,
    const Tensor& x,
    TransformerLayer& layer,
    int pos,
    int dim,
    int hidden_dim,
    int n_heads
) {
    // Garante que a saída tenha o tamanho correto
    if (out.data.size() < (size_t)dim) out.data.resize(dim);

    Tensor normed(1, dim);
    Tensor attn_out(1, dim);
    Tensor h(1, dim);
    Tensor ffn_out(1, dim);

    // Attention
    rmsnorm(normed, x, layer.norm_attn);
    attention_forward(attn_out, normed, layer.wq, layer.wk, layer.wv, layer.wo, layer.cache, pos, dim, n_heads);
    
    for (int i = 0; i < dim; i++) h.data[i] = x.data[i] + attn_out.data[i];
    
    // FFN
    rmsnorm(normed, h, layer.norm_ffn);
    ffn_forward(ffn_out, normed, layer.w1, layer.w2, layer.w3, dim, hidden_dim);
    
    for (int i = 0; i < dim; i++) out.data[i] = h.data[i] + ffn_out.data[i];
}