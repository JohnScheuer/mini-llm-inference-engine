#pragma once
#include <immintrin.h>
#include "tensor_int8.h"

inline float dot_product_q8_0_int8(
    const BlockQ8_0* W,
    const BlockQ8_0* X,
    int K)
{
    const int num_blocks = K / QK8_0;
    float result = 0.0f;

    __m256i ones = _mm256_set1_epi16(1);

    for (int b = 0; b < num_blocks; ++b)
    {
        const int8_t* wq = W[b].qs;
        const int8_t* xq = X[b].qs;

        float scale = W[b].d * X[b].d;

        // converter ativação para uint8 (add 128)
        __m256i x_signed = _mm256_loadu_si256((__m256i*)xq);
        __m256i offset = _mm256_set1_epi8(0x80);
        __m256i x_u8 = _mm256_add_epi8(x_signed, offset);

        __m256i w_s8 = _mm256_loadu_si256((__m256i*)wq);

        __m256i dot16 = _mm256_maddubs_epi16(x_u8, w_s8);
        __m256i dot32 = _mm256_madd_epi16(dot16, ones);

        __m256 dotf = _mm256_cvtepi32_ps(dot32);
        dotf = _mm256_mul_ps(dotf, _mm256_set1_ps(scale));

        __m128 low = _mm256_castps256_ps128(dotf);
        __m128 high = _mm256_extractf128_ps(dotf, 1);
        __m128 sum = _mm_add_ps(low, high);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);

        result += _mm_cvtss_f32(sum);
    }

    return result;
}