#include "layers.h"
#include <cmath>
#include <cstdio>
#include <vector>
#include <omp.h>
#include <algorithm>

// RMSNorm
void rmsnorm(Tensor& out, const Tensor& x, const Tensor& gamma, float eps) {
    float norm = 0.0f;
    int dim = x.cols;
    for (int i = 0; i < dim; i++) {
        norm += x.data[i] * x.data[i];
    }
    norm = sqrtf(norm / dim + eps);
    for (int i = 0; i < dim; i++) {
        out.data[i] = (x.data[i] / norm) * gamma.data[i];
    }
}

// RoPE
void apply_rope(Tensor& x, int pos, int dim) {
    for (int i = 0; i < dim; i += 2) {
        float freq = 1.0f / powf(10000.0f, (float)i / dim);
        float val = pos * freq;
        float cos_val = cosf(val);
        float sin_val = sinf(val);

        float x0 = x.data[i];
        float x1 = x.data[i + 1];

        x.data[i]     = x0 * cos_val - x1 * sin_val;
        x.data[i + 1] = x0 * sin_val + x1 * cos_val;
    }
}

// MatMul
void matmul(Tensor& out, const Tensor& a, const Tensor& b) {
    #pragma omp parallel for
    for (int c = 0; c < out.cols; c++) {
        float sum = 0.0f;
        for (int k = 0; k < a.cols; k++) {
            sum += a.at(0, k) * b.at(k, c);
        }
        out.at(0, c) = sum;
    }
}

// Softmax
void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }

    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// Multi-Head Attention com KV-Cache
void attention_forward(
    Tensor& out, const Tensor& x,
    const Tensor& wq, const Tensor& wk, const Tensor& wv, const Tensor& wo,
    KVCache& cache, int pos, int dim, int n_heads
) {
    int head_dim = dim / n_heads;
    float scale = 1.0f / sqrtf((float)head_dim);

    Tensor q(1, dim);
    Tensor k(1, dim);
    Tensor v(1, dim);

    matmul(q, x, wq);
    matmul(k, x, wk);
    matmul(v, x, wv);

    // Aplicar RoPE em cada cabeça
    for (int h = 0; h < n_heads; h++) {
        Tensor q_head(1, head_dim);
        Tensor k_head(1, head_dim);
        
        for (int i = 0; i < head_dim; i++) {
            q_head.data[i] = q.data[h * head_dim + i];
            k_head.data[i] = k.data[h * head_dim + i];
        }
        
        apply_rope(q_head, pos, head_dim);
        apply_rope(k_head, pos, head_dim);
        
        for (int i = 0; i < head_dim; i++) {
            q.data[h * head_dim + i] = q_head.data[i];
            k.data[h * head_dim + i] = k_head.data[i];
        }
    }

    // Salvar K e V no cache
    for (int i = 0; i < dim; i++) {
        cache.k_cache[pos * dim + i] = k.data[i];
        cache.v_cache[pos * dim + i] = v.data[i];
    }

    // Attention por cabeça
    Tensor attn_out(1, dim);
    
    #pragma omp parallel for
    for (int h = 0; h < n_heads; h++) {
        std::vector<float> scores(pos + 1, 0.0f);
        
        for (int t = 0; t <= pos; t++) {
            float score = 0.0f;
            for (int i = 0; i < head_dim; i++) {
                score += q.data[h * head_dim + i] * 
                         cache.k_cache[t * dim + h * head_dim + i];
            }
            scores[t] = score * scale;
        }

        softmax(scores.data(), pos + 1);

        for (int i = 0; i < head_dim; i++) {
            float sum = 0.0f;
            for (int t = 0; t <= pos; t++) {
                sum += scores[t] * cache.v_cache[t * dim + h * head_dim + i];
            }
            attn_out.data[h * head_dim + i] = sum;
        }
    }

    matmul(out, attn_out, wo);
}

// SiLU
float silu(float x) {
    return x / (1.0f + expf(-x));
}

// FFN com SwiGLU
void ffn_forward(
    Tensor& out, const Tensor& x,
    const Tensor& w1, const Tensor& w2, const Tensor& w3,
    int dim, int hidden_dim
) {
    Tensor gate(1, hidden_dim);
    matmul(gate, x, w1);
    
    #pragma omp parallel for
    for (int i = 0; i < hidden_dim; i++) {
        gate.data[i] = silu(gate.data[i]);
    }

    Tensor up(1, hidden_dim);
    matmul(up, x, w3);

    Tensor combined(1, hidden_dim);
    #pragma omp parallel for
    for (int i = 0; i < hidden_dim; i++) {
        combined.data[i] = gate.data[i] * up.data[i];
    }

    matmul(out, combined, w2);
}
// MatMul ingênua genérica - BASELINE
// Loop triplo clássico, ordem i-j-k (RUIM pra cache, mas é o padrão)
void matmul_naive(Tensor& C, const Tensor& A, const Tensor& B) {
    int M = A.rows;
    int K = A.cols;
    int N = B.cols;
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A.at(i, k) * B.at(k, j);
            }
            C.at(i, j) = sum;
        }
    }
}
// =====================================================
//  MatMul: Versões otimizadas
// =====================================================

// V1: Loop reordering (i-k-j) — cache-friendly por usar acessos sequenciais
// Ganho esperado: 2-5x vs naive
void matmul_ikj(Tensor& C, const Tensor& A, const Tensor& B) {
    int M = A.rows;
    int K = A.cols;
    int N = B.cols;
    
    // Zera C primeiro (porque vamos somar)
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C.at(i, j) = 0.0f;
        }
    }
    
    // Ordem i-k-j: B[k][j] é acessada sequencialmente no loop interno
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = A.at(i, k);  // carrega em registrador
            const float* B_row = &B.data[k * N];   // ponteiro pra linha
            float* C_row = &C.data[i * N];         // ponteiro pra linha
            
            // Loop interno: tudo sequencial em memória ✨
            for (int j = 0; j < N; j++) {
                C_row[j] += a_ik * B_row[j];
            }
        }
    }
}

// V2: Cache blocking (tiling)
// Divide a matriz em blocos TILE x TILE que cabem no L1 cache
// Ganho esperado: 3-7x vs naive
void matmul_tiled(Tensor& C, const Tensor& A, const Tensor& B, int tile_size) {
    int M = A.rows;
    int K = A.cols;
    int N = B.cols;
    int T = tile_size;
    
    // Zera C
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C.at(i, j) = 0.0f;
        }
    }
    
    // Loop sobre tiles
    #pragma omp parallel for collapse(2)
    for (int i0 = 0; i0 < M; i0 += T) {
        for (int j0 = 0; j0 < N; j0 += T) {
            for (int k0 = 0; k0 < K; k0 += T) {
                // Limites do tile atual (cuida das bordas)
                int i_max = std::min(i0 + T, M);
                int j_max = std::min(j0 + T, N);
                int k_max = std::min(k0 + T, K);
                
                // Multiplicação dentro do tile (ordem ijk clássica, mas pequena)
                for (int i = i0; i < i_max; i++) {
                    for (int j = j0; j < j_max; j++) {
                        float sum = C.at(i, j);
                        for (int k = k0; k < k_max; k++) {
                            sum += A.at(i, k) * B.at(k, j);
                        }
                        C.at(i, j) = sum;
                    }
                }
            }
        }
    }
}

// V3: Tiling + i-k-j combinado — versão "topo de linha"
// Ganho esperado: 5-10x vs naive
void matmul_tiled_ikj(Tensor& C, const Tensor& A, const Tensor& B, int tile_size) {
    int M = A.rows;
    int K = A.cols;
    int N = B.cols;
    int T = tile_size;
    
    // Zera C
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C.at(i, j) = 0.0f;
        }
    }
    
    // Tiling no nível externo
    #pragma omp parallel for collapse(2)
    for (int i0 = 0; i0 < M; i0 += T) {
        for (int j0 = 0; j0 < N; j0 += T) {
            for (int k0 = 0; k0 < K; k0 += T) {
                int i_max = std::min(i0 + T, M);
                int j_max = std::min(j0 + T, N);
                int k_max = std::min(k0 + T, K);
                
                // Dentro do tile: ordem i-k-j (cache-friendly)
                for (int i = i0; i < i_max; i++) {
                    for (int k = k0; k < k_max; k++) {
                        float a_ik = A.at(i, k);
                        float* C_row = &C.data[i * N];
                        const float* B_row = &B.data[k * N];
                        
                        for (int j = j0; j < j_max; j++) {
                            C_row[j] += a_ik * B_row[j];
                        }
                    }
                }
            }
        }
    }
}
// =====================================================
//  MatMul: Versões com Loop Unrolling
// =====================================================

// V4: i-k-j com unroll 4x (sem tiling)
// Processa 4 colunas de C por iteração interna
void matmul_ikj_unroll4(Tensor& C, const Tensor& A, const Tensor& B) {
    int M = A.rows;
    int K = A.cols;
    int N = B.cols;
    
    // Zera C
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C.at(i, j) = 0.0f;
        }
    }
    
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = A.at(i, k);
            const float* B_row = &B.data[k * N];
            float* C_row = &C.data[i * N];
            
            // Loop principal: processa 4 elementos por iteração
            int j = 0;
            for (; j + 4 <= N; j += 4) {
                C_row[j]     += a_ik * B_row[j];
                C_row[j + 1] += a_ik * B_row[j + 1];
                C_row[j + 2] += a_ik * B_row[j + 2];
                C_row[j + 3] += a_ik * B_row[j + 3];
            }
            
            // Resto (se N não for múltiplo de 4)
            for (; j < N; j++) {
                C_row[j] += a_ik * B_row[j];
            }
        }
    }
}

// V5: tiled + i-k-j + unroll 4x
void matmul_tiled_unroll4(Tensor& C, const Tensor& A, const Tensor& B, int tile_size) {
    int M = A.rows;
    int K = A.cols;
    int N = B.cols;
    int T = tile_size;
    
    // Zera C
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C.at(i, j) = 0.0f;
        }
    }
    
    #pragma omp parallel for collapse(2)
    for (int i0 = 0; i0 < M; i0 += T) {
        for (int j0 = 0; j0 < N; j0 += T) {
            for (int k0 = 0; k0 < K; k0 += T) {
                int i_max = std::min(i0 + T, M);
                int j_max = std::min(j0 + T, N);
                int k_max = std::min(k0 + T, K);
                
                for (int i = i0; i < i_max; i++) {
                    for (int k = k0; k < k_max; k++) {
                        float a_ik = A.at(i, k);
                        const float* B_row = &B.data[k * N];
                        float* C_row = &C.data[i * N];
                        
                        // Unroll 4x dentro do tile
                        int j = j0;
                        for (; j + 4 <= j_max; j += 4) {
                            C_row[j]     += a_ik * B_row[j];
                            C_row[j + 1] += a_ik * B_row[j + 1];
                            C_row[j + 2] += a_ik * B_row[j + 2];
                            C_row[j + 3] += a_ik * B_row[j + 3];
                        }
                        
                        // Resto
                        for (; j < j_max; j++) {
                            C_row[j] += a_ik * B_row[j];
                        }
                    }
                }
            }
        }
    }
}

// V6: tiled + i-k-j + unroll 8x com register blocking
// Versão mais agressiva: usa 8 registradores explícitos
void matmul_tiled_unroll8(Tensor& C, const Tensor& A, const Tensor& B, int tile_size) {
    int M = A.rows;
    int K = A.cols;
    int N = B.cols;
    int T = tile_size;
    
    // Zera C
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C.at(i, j) = 0.0f;
        }
    }
    
    #pragma omp parallel for collapse(2)
    for (int i0 = 0; i0 < M; i0 += T) {
        for (int j0 = 0; j0 < N; j0 += T) {
            for (int k0 = 0; k0 < K; k0 += T) {
                int i_max = std::min(i0 + T, M);
                int j_max = std::min(j0 + T, N);
                int k_max = std::min(k0 + T, K);
                
                for (int i = i0; i < i_max; i++) {
                    float* __restrict__ C_row = &C.data[i * N];
                    
                    // Unroll 8x no eixo J
                    int j = j0;
                    for (; j + 8 <= j_max; j += 8) {
                        // Carrega 8 valores de C em "registradores" (variáveis locais)
                        float c0 = C_row[j];
                        float c1 = C_row[j + 1];
                        float c2 = C_row[j + 2];
                        float c3 = C_row[j + 3];
                        float c4 = C_row[j + 4];
                        float c5 = C_row[j + 5];
                        float c6 = C_row[j + 6];
                        float c7 = C_row[j + 7];
                        
                        // Loop em K acumula tudo nos registradores
                        for (int k = k0; k < k_max; k++) {
                            float a_ik = A.at(i, k);
                            const float* B_row = &B.data[k * N + j];
                            
                            c0 += a_ik * B_row[0];
                            c1 += a_ik * B_row[1];
                            c2 += a_ik * B_row[2];
                            c3 += a_ik * B_row[3];
                            c4 += a_ik * B_row[4];
                            c5 += a_ik * B_row[5];
                            c6 += a_ik * B_row[6];
                            c7 += a_ik * B_row[7];
                        }
                        
                        // Escreve de volta na memória (1 vez só!)
                        C_row[j]     = c0;
                        C_row[j + 1] = c1;
                        C_row[j + 2] = c2;
                        C_row[j + 3] = c3;
                        C_row[j + 4] = c4;
                        C_row[j + 5] = c5;
                        C_row[j + 6] = c6;
                        C_row[j + 7] = c7;
                    }
                    
                    // Resto (j não múltiplo de 8)
                    for (; j < j_max; j++) {
                        float sum = C_row[j];
                        for (int k = k0; k < k_max; k++) {
                            sum += A.at(i, k) * B.at(k, j);
                        }
                        C_row[j] = sum;
                    }
                }
            }
        }
    }
}