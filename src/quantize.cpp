#include "tensor_int8.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>

void quantize_row_q8_0(const float* x, BlockQ8_0* y, int k) {
    int num_blocks = k / QK8_0;

    for (int i = 0; i < num_blocks; ++i) {
        const float* block = x + i * QK8_0;

        // Encontra o valor absoluto máximo com AVX2
        __m256 vmax = _mm256_setzero_ps();
        for (int j = 0; j < QK8_0; j += 8) {
            __m256 v = _mm256_loadu_ps(block + j);
            // abs: limpa o bit de sinal
            __m256 vabs = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v);
            vmax = _mm256_max_ps(vmax, vabs);
        }

        // Redução horizontal para achar o max
        __m128 hi128 = _mm256_extractf128_ps(vmax, 1);
        __m128 lo128 = _mm256_castps256_ps128(vmax);
        __m128 m128 = _mm_max_ps(lo128, hi128);
        m128 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
        m128 = _mm_max_ps(m128, _mm_shuffle_ps(m128, m128, 0x1));
        float amax = _mm_cvtss_f32(m128);

        float d = amax / 127.0f;
        y[i].d = d;

        if (d == 0.0f) {
            // Todos os valores são zero, preenche com zeros
            for (int j = 0; j < QK8_0; j++) y[i].qs[j] = 0;
            continue;
        }

        float id = 1.0f / d;
        __m256 vid = _mm256_set1_ps(id);

        // Quantiza 8 floats por vez com AVX2
        for (int j = 0; j < QK8_0; j += 8) {
            __m256 v = _mm256_loadu_ps(block + j);
            // Multiplica pelo inverso da escala
            __m256 scaled = _mm256_mul_ps(v, vid);
            // Arredonda para o inteiro mais próximo
            __m256 rounded = _mm256_round_ps(scaled,
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            // Converte para int32
            __m256i vi32 = _mm256_cvtps_epi32(rounded);

            // Extrai e armazena como int8 (pack manual)
            // AVX2 não tem pack direto para i8, então extraímos
            int32_t temp[8];
            _mm256_storeu_si256((__m256i*)temp, vi32);
            y[i].qs[j]   = (int8_t)temp[0];
            y[i].qs[j+1] = (int8_t)temp[1];
            y[i].qs[j+2] = (int8_t)temp[2];
            y[i].qs[j+3] = (int8_t)temp[3];
            y[i].qs[j+4] = (int8_t)temp[4];
            y[i].qs[j+5] = (int8_t)temp[5];
            y[i].qs[j+6] = (int8_t)temp[6];
            y[i].qs[j+7] = (int8_t)temp[7];
        }
    }
}