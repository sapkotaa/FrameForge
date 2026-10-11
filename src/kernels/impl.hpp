#pragma once
// Private: per-backend table accessors used by dispatch.cpp.
#include "kernels/dispatch.hpp"

namespace ff::detail {

const KernelTable& scalar_table();
#if defined(FF_HAVE_NEON)
const KernelTable& neon_table();
#endif
#if defined(FF_HAVE_AVX2)
const KernelTable& avx2_table();
#endif

}  // namespace ff::detail
