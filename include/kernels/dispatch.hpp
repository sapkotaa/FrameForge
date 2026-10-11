#pragma once
// Runtime-dispatched SIMD kernels.
//
// Every kernel exists as a scalar reference plus one implementation per SIMD
// instruction set (NEON on arm64, AVX2 on x86_64). The backend is chosen once
// at startup from the CPU's capabilities; tests force each backend in turn and
// require bit-exact agreement with the scalar reference.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ff {

enum class Isa { Scalar = 0, Neon = 1, Avx2 = 2 };

const char* to_string(Isa isa) noexcept;

struct KernelTable {
    Isa isa;

    // Sum of absolute differences / squared differences over a w x h block of
    // 8-bit samples. Strides are in bytes. Any w >= 0, h >= 0 (SIMD tails are
    // handled inside the kernel).
    uint64_t (*sad)(const uint8_t* a, ptrdiff_t a_stride, const uint8_t* b, ptrdiff_t b_stride, int w, int h);
    uint64_t (*ssd)(const uint8_t* a, ptrdiff_t a_stride, const uint8_t* b, ptrdiff_t b_stride, int w, int h);

    // Sum of all samples in a plane (luma energy / mean brightness).
    uint64_t (*sum)(const uint8_t* p, ptrdiff_t stride, int w, int h);

    // dst[i] = (a[i] * (256 - wb) + b[i] * wb + 128) >> 8 for wb in [0, 255].
    // Vertical pass of the bilinear resizer.
    void (*blend_rows)(const uint8_t* a, const uint8_t* b, uint8_t* dst, int n, int wb);

    // dst[i] = (r0[2i] + r0[2i+1] + r1[2i] + r1[2i+1] + 2) >> 2 for i < n_out.
    // 2x2 box filter used to build the downscale pyramid.
    void (*downscale2x_row)(const uint8_t* r0, const uint8_t* r1, uint8_t* dst, int n_out);
};

// True if the backend was compiled in AND the running CPU can execute it.
bool isa_supported(Isa isa) noexcept;
std::vector<Isa> supported_isas();
Isa best_isa() noexcept;

// The active table. Defaults to best_isa(); the FRAMEFORGE_ISA environment
// variable (scalar|neon|avx2) overrides it.
const KernelTable& kernels() noexcept;

// A specific backend's table. Throws std::invalid_argument if unsupported.
const KernelTable& kernels_for(Isa isa);

// Switch the active backend (benchmarks, tests). Throws if unsupported.
void set_active_isa(Isa isa);

}  // namespace ff
