#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "model.h"

//////////////////////////////////////////////////////////////
// Estrutura Quantizada
//////////////////////////////////////////////////////////////
struct QuantizedMatrix {
    std::vector<int8_t> weight;
    std::vector<float> scales;
    int out_dim;
    int in_dim;
};

//////////////////////////////////////////////////////////////
// Quantização per-row
//////////////////////////////////////////////////////////////
QuantizedMatrix quantize_per_row(
    const std::vector<float>& W,
    int out_dim,
    int in_dim)
{
    QuantizedMatrix qm;
    qm.out_dim = out_dim;
    qm.in_dim = in_dim;

    qm.weight.resize(out_dim * in_dim);
    qm.scales.resize(out_dim);

    for(int i = 0; i < out_dim; i++)
    {
        float max_v = 0.f;

        for(int j = 0; j < in_dim; j++)
            max_v = std::max(max_v,
                std::abs(W[i*in_dim + j]));

        float scale = max_v / 127.f;
        if(scale == 0.f) scale = 1e-8f;

        qm.scales[i] = scale;

        for(int j = 0; j < in_dim; j++)
        {
            int q = std::roundf(W[i*in_dim + j] / scale);
            q = std::max(-127, std::min(127, q));
            qm.weight[i*in_dim + j] = (int8_t)q;
        }
    }

    return qm;
}

//////////////////////////////////////////////////////////////
// RoPE
//////////////////////////////////////////////////////////////
void apply_rope_layer(std::vector<float>& vec,
                      int n_heads,
                      int head_dim,
                      int pos)
{
    for(int h=0; h<n_heads; h++) {
        for(int i=0; i<head_dim/2; i++) {

            int idx0 = h*head_dim + 2*i;
            int idx1 = h*head_dim + 2*i + 1;

            float theta = powf(10000.f, -2.f*i/head_dim);
            float angle = pos * theta;

            float c = cosf(angle);
            float s = sinf(angle);

            float v0 = vec[idx0];
            float v1 = vec[idx1];

            vec[idx0] = v0*c - v1*s;
            vec[idx1] = v0*s + v1*c;
        }
    }
}

//////////////////////////////////////////////////////////////
// SiLU
//////////////////////////////////////////////////////////////
void silu(std::vector<float>& x)
{
    for(auto& v : x)
        v = v / (1.0f + expf(-v));
}

//////////////////////////////////////////////////////////////
// MAIN
//////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    std::cout << "--- [TRANSFORMER 100% PRE-QUANTIZADO] ---\n";

    if(argc < 2) {
        std::cout << "Uso: ./tinyllama_int8 model.safetensors\n";
        return 0;
    }

    Model m;
    m.dim = 2048;
    m.hidden_dim = 5632;
    m.n_layers = 22;
    m.n_heads = 32;
    m.vocab_size = 32000;

    int n_kv_heads = 4;
    int head_dim = m.dim / m.n_heads;
    int kv_dim = n_kv_heads * head_dim;
    int kv_mul = m.n_heads / n_kv_heads;

    m.layers.resize(m.n_layers);

    load_safetensors_improved(argv[1], m);
    std::cout << "[Loader] ✓ Modelo carregado.\n";

    ////////////////////////////////////////////////////////////
    // ✅ PRÉ‑QUANTIZAR TRANSFORMER INTEIRO
    ////////////////////////////////////////////////////////////

    std::vector<QuantizedMatrix> wq_layers(m.n_layers);
    std::vector<QuantizedMatrix> wk_layers(m.n_layers);
    std::vector<QuantizedMatrix> wv_layers(m.n_layers);
    std::vector<QuantizedMatrix> wo_layers(m.n_layers);
    std::vector<QuantizedMatrix> w1_layers(m.n_layers);
    std::vector<QuantizedMatrix> w3_layers(m.n_layers);
    std::vector<QuantizedMatrix> w2_layers(m.n_layers);

    for(int l=0; l<m.n_layers; l++)
    {
        wq_layers[l] = quantize_per_row(m.layers[l].wq.data,
                                        m.dim, m.dim);

        wk_layers[l] = quantize_per_row(m.layers[l].wk.data,
                                        kv_dim, m.dim);

        wv_layers[l] = quantize_per_row(m.layers[l].wv.data,
                                        kv_dim, m.dim);

        wo_layers[l] = quantize_per_row(m.layers[l].wo.data,
                                        m.dim, m.dim);

        w1_layers[l] = quantize_per_row(m.layers[l].w1.data,
                                        m.hidden_dim, m.dim);

        w3_layers[l] = quantize_per_row(m.layers[l].w3.data,
                                        m.hidden_dim, m.dim);

        w2_layers[l] = quantize_per_row(m.layers[l].w2.data,
                                        m.dim, m.hidden_dim);
    }

    ////////////////////////////////////////////////////////////
    // Embedding
    ////////////////////////////////////////////////////////////

    int token = 15043;

    std::vector<float> x_vec(m.dim);
    for(int i=0;i<m.dim;i++)
        x_vec[i] =
            (float)m.token_embedding.data[token*m.dim + i];

    ////////////////////////////////////////////////////////////
    // Forward
    ////////////////////////////////////////////////////////////

    for(int l=0; l<m.n_layers; l++)
    {
        auto& layer = m.layers[l];
        auto& wq = wq_layers[l];
        auto& wk = wk_layers[l];
        auto& wv = wv_layers[l];
        auto& wo = wo_layers[l];
        auto& w1q = w1_layers[l];
        auto& w3q = w3_layers[l];
        auto& w2q = w2_layers[l];

        std::vector<float> residual1 = x_vec;

        // RMSNorm Attn
        std::vector<float> x_norm(m.dim);

        float rsum=0;
        for(int i=0;i<m.dim;i++)
            rsum+=x_vec[i]*x_vec[i];

        float rms=1.0f/sqrtf(rsum/m.dim+1e-5f);

        for(int i=0;i<m.dim;i++)
            x_norm[i]=x_vec[i]*rms*
                      (float)layer.norm_attn.data[i];

        std::vector<float> q(m.dim,0.f);
        std::vector<float> k(kv_dim,0.f);
        std::vector<float> v(kv_dim,0.f);

        // Q
        for(int i=0;i<m.dim;i++){
            float acc=0;
            for(int j=0;j<m.dim;j++)
                acc += (float)wq.weight[i*m.dim+j] *
                       x_norm[j];
            q[i]=acc*wq.scales[i];
        }

        // K
        for(int i=0;i<kv_dim;i++){
            float acc=0;
            for(int j=0;j<m.dim;j++)
                acc += (float)wk.weight[i*m.dim+j] *
                       x_norm[j];
            k[i]=acc*wk.scales[i];
        }

        // V
        for(int i=0;i<kv_dim;i++){
            float acc=0;
            for(int j=0;j<m.dim;j++)
                acc += (float)wv.weight[i*m.dim+j] *
                       x_norm[j];
            v[i]=acc*wv.scales[i];
        }

        apply_rope_layer(q,m.n_heads,head_dim,l);
        apply_rope_layer(k,n_kv_heads,head_dim,l);

        std::vector<float> attn_out(m.dim);

        for(int h=0;h<m.n_heads;h++){
            int kv_h=h/kv_mul;
            for(int d=0;d<head_dim;d++)
                attn_out[h*head_dim+d]=
                    v[kv_h*head_dim+d];
        }

        // Wo
        std::vector<float> o_proj(m.dim,0.f);

        for(int i=0;i<m.dim;i++){
            float acc=0;
            for(int j=0;j<m.dim;j++)
                acc += (float)wo.weight[i*m.dim+j] *
                       attn_out[j];
            o_proj[i]=acc*wo.scales[i];
        }

        for(int i=0;i<m.dim;i++)
            x_vec[i]=residual1[i]+o_proj[i];

        ////////////////////////////////////////////////////////////
        // ✅ FFN 100% PRÉ‑QUANTIZADO
        ////////////////////////////////////////////////////////////

        std::vector<float> residual2=x_vec;
        std::vector<float> x_norm2(m.dim);

        float s2=0;
        for(int i=0;i<m.dim;i++)
            s2+=x_vec[i]*x_vec[i];

        float rms2=1.0f/sqrtf(s2/m.dim+1e-5f);

        for(int i=0;i<m.dim;i++)
            x_norm2[i]=x_vec[i]*rms2*
                       (float)layer.norm_ffn.data[i];

        std::vector<float> w1(m.hidden_dim,0.f);
        std::vector<float> w3(m.hidden_dim,0.f);

        for(int i=0;i<m.hidden_dim;i++){
            float acc1=0;
            float acc3=0;

            for(int j=0;j<m.dim;j++){
                acc1 += (float)w1q.weight[i*m.dim+j] *
                        x_norm2[j];
                acc3 += (float)w3q.weight[i*m.dim+j] *
                        x_norm2[j];
            }

            w1[i]=acc1*w1q.scales[i];
            w3[i]=acc3*w3q.scales[i];
        }

        silu(w1);

        for(int i=0;i<m.hidden_dim;i++)
            w1[i]*=w3[i];

        std::vector<float> w2_out(m.dim,0.f);

        for(int i=0;i<m.dim;i++){
            float acc=0;
            for(int j=0;j<m.hidden_dim;j++)
                acc += (float)w2q.weight[i*m.hidden_dim+j] *
                       w1[j];
            w2_out[i]=acc*w2q.scales[i];
        }

        for(int i=0;i<m.dim;i++)
            x_vec[i]=residual2[i]+w2_out[i];

        if(l==0)
            std::cout<<"[DEBUG] L0 x[0]: "<<x_vec[0]<<std::endl;
    }

    ////////////////////////////////////////////////////////////
    // Final Norm + Logits
    ////////////////////////////////////////////////////////////

    float fs=0;
    for(int i=0;i<m.dim;i++)
        fs+=x_vec[i]*x_vec[i];

    float frms=1.0f/sqrtf(fs/m.dim+1e-5f);

    for(int i=0;i<m.dim;i++)
        x_vec[i]=x_vec[i]*frms*
                 (float)m.norm_final.data[i];

    std::cout<<"\nPrimeiros 10 Logits:\n";

    for(int i=0;i<10;i++){
        float logit=0;
        for(int j=0;j<m.dim;j++)
            logit+=m.lm_head.data[i*m.dim+j]*x_vec[j];
        std::cout<<logit<<" ";
    }

    std::cout<<std::endl;

    return 0;
}