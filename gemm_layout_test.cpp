#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>

//////////////////////////////////////////////////////////////
// Row-major GEMM (M x K) × (K)
//////////////////////////////////////////////////////////////
void gemm_row_major(
    const std::vector<float>& W,
    const std::vector<float>& x,
    std::vector<float>& y,
    int M,
    int K)
{
    for(int i=0;i<M;i++)
    {
        float acc = 0.f;

        for(int j=0;j<K;j++)
            acc += W[i*K + j] * x[j];

        y[i] = acc;
    }
}

//////////////////////////////////////////////////////////////
// Convert Row → Col major
//////////////////////////////////////////////////////////////
std::vector<float> to_col_major(
    const std::vector<float>& W,
    int M,
    int K)
{
    std::vector<float> W_col(M*K);

    for(int i=0;i<M;i++)
        for(int j=0;j<K;j++)
            W_col[j*M + i] = W[i*K + j];

    return W_col;
}

//////////////////////////////////////////////////////////////
// Col-major GEMM (M x K) × (K)
//////////////////////////////////////////////////////////////
void gemm_col_major(
    const std::vector<float>& W_col,
    const std::vector<float>& x,
    std::vector<float>& y,
    int M,
    int K)
{
    for(int i=0;i<M;i++)
    {
        float acc = 0.f;

        for(int j=0;j<K;j++)
            acc += W_col[j*M + i] * x[j];

        y[i] = acc;
    }
}

//////////////////////////////////////////////////////////////
// MAIN
//////////////////////////////////////////////////////////////
int main()
{
    const int M = 128;
    const int K = 128;

    std::cout << "--- GEMM Layout Test ---\n";

    std::vector<float> W(M*K);
    std::vector<float> x(K);
    std::vector<float> y_row(M);
    std::vector<float> y_col(M);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    for(int i=0;i<M*K;i++)
        W[i] = dist(rng);

    for(int i=0;i<K;i++)
        x[i] = dist(rng);

    gemm_row_major(W, x, y_row, M, K);

    auto W_col = to_col_major(W, M, K);

    gemm_col_major(W_col, x, y_col, M, K);

    float max_diff = 0.f;
    float mean_diff = 0.f;

    for(int i=0;i<M;i++)
    {
        float diff = std::abs(y_row[i] - y_col[i]);
        max_diff = std::max(max_diff, diff);
        mean_diff += diff;
    }

    mean_diff /= M;

    std::cout << "Max abs diff:  " << max_diff << "\n";
    std::cout << "Mean abs diff: " << mean_diff << "\n";

    return 0;
}