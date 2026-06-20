#include <vector>
#include <iostream>
#include <cmath>
#include <cstring>
#include "model.h"

//////////////////////////////////////////////////////////////
// RoPE
//////////////////////////////////////////////////////////////

void apply_rope(std::vector<float>& q,
                std::vector<float>& k,
                int dim,
                int n_heads,
                int pos)
{
    int head_dim = dim / n_heads;

    for(int h=0; h<n_heads; h++)
    {
        for(int i=0; i<head_dim/2; i++)
        {
            int idx0 = h*head_dim + 2*i;
            int idx1 = h*head_dim + 2*i + 1;

            float theta = powf(10000.f, -2.f*i/head_dim);
            float angle = pos * theta;

            float c = cosf(angle);
            float s = sinf(angle);

            float q0 = q[idx0];
            float q1 = q[idx1];

            float k0 = k[idx0];
            float k1 = k[idx1];

            q[idx0] = q0*c - q1*s;
            q[idx1] = q0*s + q1*c;

            k[idx0] = k0*c - k1*s;
            k[idx1] = k0*s + k1*c;
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
    if (argc < 2) {
        std::cout << "Usage: ./tinyllama_fp16_minimal model\n";
        return 0;
    }

    std::string model_path = argv[1];

    Model m;
    m.dim = 2048;
    m.hidden_dim = 5632;
    m.n_layers = 22;
    m.n_heads = 32;
    m.vocab_size = 32000;
    m.max_seq_len = 2048;
    m.layers.resize(m.n_layers);

    load_safetensors_improved(model_path, m);

    int token = 15043; // "Hello"

    ////////////////////////////////////////////////////////////
    // Embedding
    ////////////////////////////////////////////////////////////

    std::vector<float> x(m.dim, 0.f);
    for(int i=0;i<m.dim;i++)
        x[i] = m.token_embedding.data[token*m.dim + i];

    ////////////////////////////////////////////////////////////
    // Forward
    ////////////////////////////////////////////////////////////

    for(int l=0; l<m.n_layers; l++)
    {
        auto& layer = m.layers[l];

        std::vector<float> residual1 = x;

        // ----- RMSNorm Attn -----
        std::vector<float> x_norm(m.dim, 0.f);
        {
            float sum = 0.f;
            for(int i=0;i<m.dim;i++)
                sum += x[i]*x[i];

            float rms = 1.0f / sqrtf(sum/m.dim + 1e-5f);

            for(int i=0;i<m.dim;i++)
                x_norm[i] = x[i] * rms * layer.norm_attn.data[i];
        }

        // ----- QKV -----
        std::vector<float> q(m.dim, 0.f);
        std::vector<float> k(m.dim, 0.f);
        std::vector<float> v(m.dim, 0.f);

        for(int i=0;i<m.dim;i++)
            for(int j=0;j<m.dim;j++)
            {
                q[i] += layer.wq.data[i*m.dim + j] * x_norm[j];
                k[i] += layer.wk.data[i*m.dim + j] * x_norm[j];
                v[i] += layer.wv.data[i*m.dim + j] * x_norm[j];
            }

        apply_rope(q, k, m.dim, m.n_heads, l);

        // ----- Attention (seq_len=1) -----
        std::vector<float> attn(m.dim, 0.f);
        int head_dim = m.dim / m.n_heads;

        for(int h=0; h<m.n_heads; h++)
        {
            float score = 0.f;
            for(int d=0; d<head_dim; d++)
                score += q[h*head_dim+d] * k[h*head_dim+d];

            score /= sqrtf(head_dim);

            for(int d=0; d<head_dim; d++)
                attn[h*head_dim+d] = v[h*head_dim+d];
        }

        // ----- O projection -----
        std::vector<float> o(m.dim, 0.f);
        for(int i=0;i<m.dim;i++)
            for(int j=0;j<m.dim;j++)
                o[i] += layer.wo.data[i*m.dim + j] * attn[j];

        for(int i=0;i<m.dim;i++)
            x[i] = residual1[i] + o[i];

        std::vector<float> residual2 = x;

        // ----- RMSNorm FFN -----
        std::vector<float> x_norm2(m.dim, 0.f);
        {
            float sum = 0.f;
            for(int i=0;i<m.dim;i++)
                sum += x[i]*x[i];

            float rms = 1.0f / sqrtf(sum/m.dim + 1e-5f);

            for(int i=0;i<m.dim;i++)
                x_norm2[i] = x[i] * rms * layer.norm_ffn.data[i];
        }

        // ----- W1 & W3 -----
        std::vector<float> w1(m.hidden_dim, 0.f);
        std::vector<float> w3(m.hidden_dim, 0.f);

        for(int i=0;i<m.hidden_dim;i++)
            for(int j=0;j<m.dim;j++)
            {
                w1[i] += layer.w1.data[i*m.dim + j] * x_norm2[j];
                w3[i] += layer.w3.data[i*m.dim + j] * x_norm2[j];
            }

        silu(w1);

        for(int i=0;i<m.hidden_dim;i++)
            w1[i] *= w3[i];

        // ----- W2 -----
        std::vector<float> w2(m.dim, 0.f);
        for(int i=0;i<m.dim;i++)
            for(int j=0;j<m.hidden_dim;j++)
                w2[i] += layer.w2.data[i*m.hidden_dim + j] * w1[j];

        for(int i=0;i<m.dim;i++)
            x[i] = residual2[i] + w2[i];
    }

    ////////////////////////////////////////////////////////////
    // Final RMSNorm
    ////////////////////////////////////////////////////////////

    {
        float sum = 0.f;
        for(int i=0;i<m.dim;i++)
            sum += x[i]*x[i];

        float rms = 1.0f / sqrtf(sum/m.dim + 1e-5f);

        for(int i=0;i<m.dim;i++)
            x[i] = x[i] * rms * m.norm_final.data[i];
    }

    ////////////////////////////////////////////////////////////
    // Debug x before LM head
    ////////////////////////////////////////////////////////////

    std::cout << "\nFirst 5 x before LM head:\n";
    for(int i=0;i<5;i++)
        std::cout << x[i] << " ";
    std::cout << "\n";

    ////////////////////////////////////////////////////////////
    // LM head
    ////////////////////////////////////////////////////////////

    std::vector<float> logits(m.vocab_size, 0.f);

    for(int i=0;i<m.vocab_size;i++)
        for(int j=0;j<m.dim;j++)
            logits[i] += m.lm_head.data[i*m.dim + j] * x[j];

    std::cout << "\nFirst 10 logits:\n";
    for(int i=0;i<10;i++)
        std::cout << logits[i] << " ";
    std::cout << "\n";

    return 0;
}