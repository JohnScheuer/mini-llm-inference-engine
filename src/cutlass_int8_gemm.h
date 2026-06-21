#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/epilogue/thread/linear_combination.h"

using CutlassInt8Gemm =
    cutlass::gemm::device::Gemm<
        int8_t,
        cutlass::layout::RowMajor,      // W2 em RowMajor
        int8_t,
        cutlass::layout::ColumnMajor,   // Ativação
        cutlass::half_t,
        cutlass::layout::ColumnMajor,
        int32_t,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm75,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            cutlass::half_t,
            8,
            int32_t,
            float
        >
    >;