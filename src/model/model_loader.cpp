#include "backend/int8_tp_backend.h"
#include "model/model_weights.h"
#include "runtime/model_config.h"
#include "runtime/request.h"
#include "runtime/response.h"
#include "runtime/kv_cache.h"
#include "runtime/tp_context.h"
#include "kernels/mlp_int8.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <vector>
#include <iostream>
#include <cstring>
#include <algorithm>

namespace runtime {

////////////////////////////////////////////////////////////////
// ====================== PIMPL ===============================
////////////////////////////////////////////////////////////////

struct Int8TPBackend::Impl {

    ModelConfig config;
    TPContext tp;

    cublasHandle_t cublas_handle = nullptr;

    // Buffers de ativação (já definidos antes)
    half* d_x = nullptr;
    half* d_o = nullptr;
    half* d_w1 = nullptr;
    half* d_w3 = nullptr;
    half* d_w2 = nullptr;
    half* d_logits = nullptr;
    int8_t*  d_x_int8 = nullptr;
    float*   d_x_scale = nullptr;
    int32_t* d_out_int32 = nullptr;

    // Pesos do modelo na GPU
    struct LayerWeightsGPU {
        half* wq = nullptr;
        half* wk = nullptr;
        half* wv = nullptr;
        // Wo shardado para TP: [world_size] ponteiros
        std::vector<half*> wo_shards;
        
        // W1 em INT8 (quantizado row-wise)
        int8_t* w1_int8 = nullptr;
        float*  w1_scale = nullptr;
        
        half* w3 = nullptr;
        half* w2 = nullptr;
        half* norm_attn = nullptr;
        half* norm_ffn = nullptr;
    };

    half* token_embedding = nullptr;
    std::vector<LayerWeightsGPU> layers;
    half* norm_final = nullptr;
    half* lm_head = nullptr;

    bool initialized = false;
    static constexpr int BATCH_SIZE = 8;

    ////////////////////////////////////////////////////////////
    // Helpers de upload
    ////////////////////////////////////////////////////////////
    
    half* upload_float_to_half(const std::vector<float>& src) {
        if (src.empty()) return nullptr;
        
        std::vector<half> tmp(src.size());
        for (size_t i = 0; i < src.size(); ++i)
            tmp[i] = __float2half(src[i]);

        half* d_ptr = nullptr;
        cudaMalloc(&d_ptr, src.size() * sizeof(half));
        cudaMemcpy(d_ptr, tmp.data(), src.size() * sizeof(half), 
                   cudaMemcpyHostToDevice);
        return d_ptr;
    }

    void quantize_weight_rowwise(const std::vector<float>& src,
                                 std::vector<int8_t>& q,
                                 std::vector<float>& scale,
                                 int rows, int cols) 
    {
        q.resize(rows * cols);
        scale.resize(rows);

        for (int r = 0; r < rows; ++r) {
            float max_abs = 0.f;
            for (int c = 0; c < cols; ++c) {
                float v = std::fabs(src[c * rows + r]);  // column-major access
                if (v > max_abs) max_abs = v;
            }
            float s = max_abs > 0 ? max_abs / 127.f : 1.f;
            scale[r] = s;

            for (int c = 0; c < cols; ++c) {
                float v = src[c * rows + r] / s;
                int iv = static_cast<int>(std::round(v));
                iv = std::max(-127, std::min(127, iv));
                q[c * rows + r] = static_cast<int8_t>(iv);
            }
        }
    }

    Status allocate_buffers();
    Status upload_weights(const ModelWeights& weights);
    void run_layer(int layer_idx, cudaStream_t stream);
    void run_lm_head(cudaStream_t stream);
};

////////////////////////////////////////////////////////////////
// UPLOAD WEIGHTS (IMPLEMENTAÇÃO PRINCIPAL)
////////////////////////////////////////////////////////////////

Status Int8TPBackend::Impl::upload_weights(const ModelWeights& weights) 
{
    if (!config.is_valid()) return Status::INVALID_ARGUMENT;

    int world_size = tp.world_size;
    int local_dim = tp.enabled() ? config.dim / world_size : config.dim;

    // 1️⃣ Token Embedding
    token_embedding = upload_float_to_half(weights.token_embedding);

    // 2️⃣ LM Head e Norm Final
    lm_head = upload_float_to_half(weights.lm_head);
    norm_final = upload_float_to_half(weights.norm_final);

    // 3️⃣ Layers
    layers.resize(config.n_layers);

    for (int l = 0; l < config.n_layers; ++l) {
        const auto& src_layer = weights.layers[l];
        auto& gpu_layer = layers[l];

        // Projeções de atenção (FP16)
        gpu_layer.wq = upload_float_to_half(src_layer.wq);
        gpu_layer.wk = upload_float_to_half(src_layer.wk);
        gpu_layer.wv = upload_float_to_half(src_layer.wv);
        
        // Norms (FP16)
        gpu_layer.norm_attn = upload_float_to_half(src_layer.norm_attn);
        gpu_layer.norm_ffn = upload_float_to_half(src_layer.norm_ffn);

        // 4️⃣ Wo - Row Parallel Sharding
        // Wo original: [dim, dim] (col-major: [row][col] -> [dim * dim])
        // Shard por linhas (output dim): cada rank pega local_dim linhas
        gpu_layer.wo_shards.resize(world_size);
        
        for (int r = 0; r < world_size; ++r) {
            std::vector<float> wo_local(local_dim * config.dim);
            
            // Copiar slice de linhas [r*local_dim, (r+1)*local_dim)
            for (int local_row = 0; local_row < local_dim; ++local_row) {
                int global_row = r * local_dim + local_row;
                for (int col = 0; col < config.dim; ++col) {
                    // src: column-major [col * rows + row]
                    wo_local[local_row * config.dim + col] = 
                        src_layer.wo[col * config.dim + global_row];
                }
            }
            
            gpu_layer.wo_shards[r] = upload_float_to_half(wo_local);
        }

        // 5️⃣ W1 - Quantização INT8 Row-wise
        // W1: [hidden_dim, dim] -> quantizar por linha (hidden_dim)
        std::vector<int8_t> w1_q;
        std::vector<float>  w1_s;
        
        quantize_weight_rowwise(src_layer.w1, w1_q, w1_s, 
                               config.hidden_dim, config.dim);
        
        // Upload INT8 e scales
        cudaMalloc(&gpu_layer.w1_int8, w1_q.size());
        cudaMemcpy(gpu_layer.w1_int8, w1_q.data(), w1_q.size(), 
                   cudaMemcpyHostToDevice);
        
        cudaMalloc(&gpu_layer.w1_scale, w1_s.size() * sizeof(float));
        cudaMemcpy(gpu_layer.w1_scale, w1_s.data(), 
                   w1_s.size() * sizeof(float), cudaMemcpyHostToDevice);

        // 6️⃣ W3 e W2 (FP16)
        gpu_layer.w3 = upload_float_to_half(src_layer.w3);
        gpu_layer.w2 = upload_float_to_half(src_layer.w2);
    }

    std::cout << "[Int8TPBackend] Pesos carregados na GPU:" << std::endl;
    std::cout << "  - Layers: " << config.n_layers << std::endl;
    std::cout << "  - TP: " << (tp.enabled() ? "Sim (world_size=" + 
             std::to_string(world_size) + ")" : "Não") << std::endl;
    std::cout << "  - W1: Quantizado INT8 row-wise" << std::endl;

    return Status::OK;
}

////////////////////////////////////////////////////////////////
// RESTANTE DA IMPLEMENTAÇÃO DO BACKEND
////////////////////////////////////////////////////////////////

Status Int8TPBackend::Impl::allocate_buffers() {
    size_t dim = config.dim;
    size_t hidden = config.hidden_dim;
    size_t vocab = config.vocab_size;

    cudaMalloc(&d_x, dim * BATCH_SIZE * sizeof(half));
    cudaMalloc(&d_o, dim * BATCH_SIZE * sizeof(half));
    cudaMalloc(&d_w1, hidden * BATCH_SIZE * sizeof(half));
    cudaMalloc(&d_w3, hidden * BATCH_SIZE * sizeof(half));
    cudaMalloc(&d_w2, dim * BATCH_SIZE * sizeof(half));
    cudaMalloc(&d_logits, vocab * BATCH_SIZE * sizeof(half));

    cudaMalloc(&d_x_int8, dim * BATCH_SIZE);
    cudaMalloc(&d_x_scale, BATCH_SIZE * sizeof(float));
    cudaMalloc(&d_out_int32, hidden * BATCH_SIZE * sizeof(int32_t));

    return Status::OK;
}

void Int8TPBackend::Impl::run_layer(int layer_idx, cudaStream_t stream) {
    // Implementação similar ao que você tinha, mas usando gpu_layer.w1_int8, etc.
    // ...
}

void Int8TPBackend::Impl::run_lm_head(cudaStream_t stream) {
    // GEMM final
}

////////////////////////////////////////////////////////////////
// INTERFACE PÚBLICA
////////////////////////////////////////////////////////////////

Int8TPBackend::Int8TPBackend() : impl_(new Impl()) {}

Int8TPBackend::~Int8TPBackend() {
    shutdown();
    delete impl_;
}

Status Int8TPBackend::initialize(const ModelConfig& config,
                                const TPContext* tp_context) 
{
    impl_->config = config;
    
    if (tp_context) {
        impl_->tp = *tp_context;
    }

    if (!config.is_valid()) {
        return Status::INVALID_ARGUMENT;
    }

    if (impl_->tp.enabled() && !impl_->tp.is_valid(config.dim)) {
        return Status::INVALID_ARGUMENT;
    }

    if (cublasCreate(&impl_->cublas_handle) != CUBLAS_STATUS_SUCCESS) {
        return Status::CUDA_ERROR;
    }

    Status s = impl_->allocate_buffers();
    if (s != Status::OK) return s;

    impl_->initialized = true;
    return Status::OK;
}

// Wrapper público para upload de pesos
Status Int8TPBackend::upload_weights(const ModelWeights& weights) {
    if (!impl_ || !impl_->initialized) return Status::NOT_INITIALIZED;
    return impl_->upload_weights(weights);
}

Status Int8TPBackend::shutdown() {
    if (!impl_) return Status::OK;
    
    if (impl_->cublas_handle) {
        cublasDestroy(impl_->cublas_handle);
        impl_->cublas_handle = nullptr;
    }

    // Liberar buffers de ativação
    cudaFree(impl_->d_x); impl_->d_x = nullptr;
    cudaFree(impl_->d_o); impl_->d_o = nullptr;
    cudaFree(impl_->d_w1); impl_->d_w1 = nullptr;
    cudaFree(impl_->d_w3); impl_->d_w3 = nullptr;
    cudaFree(impl_->d_w2); impl_->d_w2 = nullptr;
    cudaFree(impl_->d_logits); impl_->d_logits = nullptr;
    cudaFree(impl_->d_x_int8); impl_->d_x_int8 = nullptr;
    cudaFree(impl_->d_x_scale); impl_->d_x_scale = nullptr;
    cudaFree(impl_->d_out_int32); impl_->d_out_int32 = nullptr;

    // Liberar pesos
    cudaFree(impl_->token_embedding);
    cudaFree(impl_->norm_final);
    cudaFree(impl_->lm_head);
    
    for (auto& layer : impl_->layers) {
        cudaFree(layer.wq);
        cudaFree(layer.wk);
        cudaFree(layer.wv);
        for (auto ptr : layer.wo_shards) cudaFree(ptr);
        cudaFree(layer.w1_int8);
        cudaFree(layer.w1_scale);
        cudaFree(layer.w3);
        cudaFree(layer.w2);
        cudaFree(layer.norm_attn);
        cudaFree(layer.norm_ffn);
    }
    impl_->layers.clear();

    impl_->initialized = false;
    return Status::OK;
}

Status Int8TPBackend::prefill(const InferenceRequest&, KVCache&, cudaStream_t) {
    return Status::UNSUPPORTED_OPERATION;
}

Status Int8TPBackend::decode_one_token(const InferenceRequest& request,
                                      KVCache&,
                                      InferenceResponse& response,
                                      cudaStream_t stream) 
{
    if (!impl_->initialized) return Status::NOT_INITIALIZED;
    if (!request.is_valid()) return Status::INVALID_ARGUMENT;

    // TODO: Embedding do token
    // TODO: Loop pelas camadas
    // TODO: Sample e preencher response
    
    response.output_tokens.push_back(0); // placeholder
    response.success = true;
    return Status::OK;
}

Status Int8TPBackend::decode_batch(const std::vector<InferenceRequest>&,
                                  std::vector<KVCache*>&,
                                  std::vector<InferenceResponse>&,
                                  cudaStream_t) 
{
    return Status::UNSUPPORTED_OPERATION;
}

const char* Int8TPBackend::name() const {
    return "Int8TPBackend";
}

} // namespace runtime