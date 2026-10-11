// AArch64 NEON backend (Apple Silicon, Graviton, ...). NEON is mandatory on
// AArch64, so there is no runtime feature probe.

#if defined(FF_HAVE_NEON)

#include <arm_neon.h>

#include <cstdlib>
#include <cstring>

#include "impl.hpp"

namespace ff::detail {
namespace {

uint64_t sad_neon(const uint8_t* a, ptrdiff_t as, const uint8_t* b, ptrdiff_t bs, int w, int h) {
    uint64_t total = 0;
    for (int y = 0; y < h; ++y) {
        const uint8_t* pa = a + y * as;
        const uint8_t* pb = b + y * bs;
        uint32x4_t acc = vdupq_n_u32(0);
        int x = 0;
        for (; x + 16 <= w; x += 16) {
            const uint8x16_t d = vabdq_u8(vld1q_u8(pa + x), vld1q_u8(pb + x));
            acc = vpadalq_u16(acc, vpaddlq_u8(d));  // u8 -> u16 pairs -> accumulate into u32
        }
        uint32_t row = vaddvq_u32(acc);
        if (x + 8 <= w) {
            row += vaddlv_u8(vabd_u8(vld1_u8(pa + x), vld1_u8(pb + x)));
            x += 8;
        }
        for (; x < w; ++x) row += static_cast<uint32_t>(std::abs(pa[x] - pb[x]));
        total += row;
    }
    return total;
}

uint64_t ssd_neon(const uint8_t* a, ptrdiff_t as, const uint8_t* b, ptrdiff_t bs, int w, int h) {
    uint64_t total = 0;
    for (int y = 0; y < h; ++y) {
        const uint8_t* pa = a + y * as;
        const uint8_t* pb = b + y * bs;
        uint32x4_t acc = vdupq_n_u32(0);
        int x = 0;
        for (; x + 16 <= w; x += 16) {
            const uint8x16_t d = vabdq_u8(vld1q_u8(pa + x), vld1q_u8(pb + x));
            acc = vpadalq_u16(acc, vmull_u8(vget_low_u8(d), vget_low_u8(d)));  // |d|^2 fits u16
            acc = vpadalq_u16(acc, vmull_high_u8(d, d));
        }
        uint64_t row = vaddvq_u32(acc);  // u32 lanes cannot overflow for rows < ~260k pixels
        for (; x < w; ++x) {
            const int d = pa[x] - pb[x];
            row += static_cast<uint64_t>(d * d);
        }
        total += row;
    }
    return total;
}

uint64_t sum_neon(const uint8_t* p, ptrdiff_t stride, int w, int h) {
    uint64_t total = 0;
    for (int y = 0; y < h; ++y) {
        const uint8_t* r = p + y * stride;
        uint32x4_t acc = vdupq_n_u32(0);
        int x = 0;
        for (; x + 16 <= w; x += 16) acc = vpadalq_u16(acc, vpaddlq_u8(vld1q_u8(r + x)));
        uint64_t s = vaddvq_u32(acc);
        for (; x < w; ++x) s += r[x];
        total += s;
    }
    return total;
}

void blend_rows_neon(const uint8_t* a, const uint8_t* b, uint8_t* dst, int n, int wb) {
    if (wb == 0) {  // weight 256 does not fit in u8; the result is just a
        std::memcpy(dst, a, static_cast<size_t>(n));
        return;
    }
    const uint8x16_t w1 = vdupq_n_u8(static_cast<uint8_t>(wb));
    const uint8x16_t w0 = vdupq_n_u8(static_cast<uint8_t>(256 - wb));
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        const uint8x16_t va = vld1q_u8(a + i);
        const uint8x16_t vb = vld1q_u8(b + i);
        uint16x8_t lo = vmull_u8(vget_low_u8(va), vget_low_u8(w0));
        lo = vmlal_u8(lo, vget_low_u8(vb), vget_low_u8(w1));
        uint16x8_t hi = vmull_high_u8(va, w0);
        hi = vmlal_high_u8(hi, vb, w1);
        // vrshrn computes (x + 128) >> 8, exactly the scalar rounding.
        vst1q_u8(dst + i, vcombine_u8(vrshrn_n_u16(lo, 8), vrshrn_n_u16(hi, 8)));
    }
    const int wa = 256 - wb;
    for (; i < n; ++i) dst[i] = static_cast<uint8_t>((a[i] * wa + b[i] * wb + 128) >> 8);
}

void downscale2x_row_neon(const uint8_t* r0, const uint8_t* r1, uint8_t* dst, int n_out) {
    int i = 0;
    for (; i + 8 <= n_out; i += 8) {
        uint16x8_t s = vpaddlq_u8(vld1q_u8(r0 + 2 * i));  // horizontal pair sums
        s = vpadalq_u8(s, vld1q_u8(r1 + 2 * i));           // + the row below
        vst1_u8(dst + i, vrshrn_n_u16(s, 2));              // (s + 2) >> 2
    }
    for (; i < n_out; ++i)
        dst[i] = static_cast<uint8_t>((r0[2 * i] + r0[2 * i + 1] + r1[2 * i] + r1[2 * i + 1] + 2) >> 2);
}

}  // namespace

const KernelTable& neon_table() {
    static const KernelTable t{Isa::Neon,  sad_neon,         ssd_neon,
                               sum_neon,   blend_rows_neon,  downscale2x_row_neon};
    return t;
}

}  // namespace ff::detail

#endif  // FF_HAVE_NEON
