#pragma once

#include <memory>
#include <vector>
#include <cstdint>

namespace runtime {

// Forward declaration para evitar incluir cuda_runtime.h
struct CUstream_st;
using cudaStream_t = CUstream_st*;

// ============================================================
// Forward declarations
// ============================================================

struct ModelConfig;
struct KVCache;
struct InferenceRequest;
struct InferenceResponse;
struct TPContext;

// ============================================================
// Status enum
// ============================================================

enum class Status {
    OK = 0,
    INVALID_ARGUMENT,
    RUNTIME_ERROR,
    CUDA_ERROR,
    OUT_OF_MEMORY,
    NOT_INITIALIZED,
    UNSUPPORTED_OPERATION
};

// ============================================================
// Backend Interface
// ============================================================

class ILLMBackend {
public:
    virtual ~ILLMBackend() = default;

    virtual Status initialize(
        const ModelConfig& config,
        const TPContext* tp_context = nullptr
    ) = 0;

    virtual Status prefill(
        const InferenceRequest& request,
        KVCache& kv_cache,
        cudaStream_t stream
    ) = 0;

    virtual Status decode_one_token(
        const InferenceRequest& request,
        KVCache& kv_cache,
        InferenceResponse& response,
        cudaStream_t stream
    ) = 0;

    virtual Status decode_batch(
        const std::vector<InferenceRequest>& requests,
        std::vector<KVCache*>& kv_caches,
        std::vector<InferenceResponse>& responses,
        cudaStream_t stream
    ) = 0;

    virtual Status shutdown() = 0;

    virtual const char* name() const = 0;
};

} // namespace runtime