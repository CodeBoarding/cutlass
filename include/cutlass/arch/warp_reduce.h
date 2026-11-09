/***************************************************************************************************
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/config.h"
#include <cuda_runtime.h>

#if defined(CUTLASS_ARCH_MMA_SM100A_ENABLED) || defined(CUTLASS_ARCH_MMA_SM100F_ENABLED) ||\
    defined(CUTLASS_ARCH_MMA_SM103A_ENABLED) || defined(CUTLASS_ARCH_MMA_SM103F_ENABLED)
#  define CUTLASS_ARCH_CREDUX_ENABLED
#endif

namespace cutlass {
namespace arch {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Reduce independent groups of Width consecutive lanes, broadcasting each group's result.
/// All 32 lanes of the warp must participate, even when Width is smaller than 32.
/// The operator must be associative; the shuffle tree order matches a down-shuffle reduction.
template <int Width, class ReductionOp>
CUTLASS_DEVICE
float warp_reduce(float value, ReductionOp op) {
  static_assert(Width > 0 && Width <= 32 && (Width & (Width - 1)) == 0,
                "Warp reduction width must be a power of two between 1 and 32");
#if defined(__CUDA_ARCH__)
  CUTLASS_PRAGMA_UNROLL
  for (int delta = Width / 2; delta > 0; delta /= 2) {
    float other = __shfl_down_sync(0xffffffff, value, delta, Width);
    value = op(value, other);
  }
  return __shfl_sync(0xffffffff, value, 0, Width);
#else
  CUTLASS_UNUSED(value);
  CUTLASS_UNUSED(op);
  CUTLASS_NOT_IMPLEMENTED();
  return 0;
#endif
}

/// Native warp-wide absolute maximum, with NaN propagation, on supported architectures.
/// All 32 lanes must participate. Callers select the shuffle fallback on other architectures.
CUTLASS_DEVICE
float redux_abs_max_nan_sync(float value) {
#if defined(CUTLASS_ARCH_CREDUX_ENABLED)
  float result;
  asm volatile("redux.sync.max.abs.NaN.f32 %0, %1, 0xffffffff;\n" : "=f"(result) : "f"(value));
  return result;
#else
  CUTLASS_UNUSED(value);
  CUTLASS_NOT_IMPLEMENTED();
  return 0;
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace cutlass
