// Scalar reference kernels. Compiled with auto-vectorization disabled (see
// CMakeLists.txt) so they are both the bit-exact reference for the SIMD
// versions and an honest "no SIMD" baseline for the benchmarks.

#include <cstdlib>
#include <cstring>

#include "impl.hpp"

namespace ff::detail {
namespace {

uint64_t sad_scalar(const uint8_t* a, ptrdiff_t as, const uint8_t* b, ptrdiff_t bs, int w, int h) {
    uint64_t total = 0;
    for (int y = 0; y < h; ++y) {
        const uint8_t* pa = a + y * as;
        const uint8_t* pb = b + y * bs;
        uint32_t row = 0;
        for (int x = 0; x < w; ++x) row += static_cast<uint32_t>(std::abs(pa[x] - pb[x]));
        total += row;
    }
    return total;
}

uint64_t ssd_scalar(const uint8_t* a, ptrdiff_t as, const uint8_t* b, ptrdiff_t bs, int w, int h) {
    uint64_t total = 0;
    for (int y = 0; y < h; ++y) {
        const uint8_t* pa = a + y * as;
        const uint8_t* pb = b + y * bs;
        uint64_t row = 0;
        for (int x = 0; x < w; ++x) {
            const int d = pa[x] - pb[x];
            row += static_cast<uint64_t>(d * d);
        }
        total += row;
    }
    return total;
}

uint64_t sum_scalar(const uint8_t* p, ptrdiff_t stride, int w, int h) {
    uint64_t total = 0;
    for (int y = 0; y < h; ++y) {
        const uint8_t* row = p + y * stride;
        uint64_t s = 0;
        for (int x = 0; x < w; ++x) s += row[x];
        total += s;
    }
    return total;
}

void blend_rows_scalar(const uint8_t* a, const uint8_t* b, uint8_t* dst, int n, int wb) {
    const int wa = 256 - wb;
    for (int i = 0; i < n; ++i) dst[i] = static_cast<uint8_t>((a[i] * wa + b[i] * wb + 128) >> 8);
}

void downscale2x_row_scalar(const uint8_t* r0, const uint8_t* r1, uint8_t* dst, int n_out) {
    for (int i = 0; i < n_out; ++i)
        dst[i] = static_cast<uint8_t>((r0[2 * i] + r0[2 * i + 1] + r1[2 * i] + r1[2 * i + 1] + 2) >> 2);
}

}  // namespace

const KernelTable& scalar_table() {
    static const KernelTable t{Isa::Scalar,        sad_scalar,         ssd_scalar,
                               sum_scalar,         blend_rows_scalar,  downscale2x_row_scalar};
    return t;
}

}  // namespace ff::detail
