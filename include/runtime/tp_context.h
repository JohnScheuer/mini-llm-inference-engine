#pragma once

#include <cstdint>

namespace runtime {

struct TPContext {

    int32_t world_size = 1;
    int32_t rank = 0;

    // Dimensão local por shard
    int32_t local_dim = 0;

    bool enabled() const {
        return world_size > 1;
    }

    bool is_valid(int32_t model_dim) const {

        if (world_size <= 0)
            return false;

        if (rank < 0 || rank >= world_size)
            return false;

        if (!enabled())
            return true;

        return (model_dim % world_size) == 0;
    }

    int32_t compute_local_dim(int32_t model_dim) const {
        return enabled() ? model_dim / world_size : model_dim;
    }
};

} // namespace runtime