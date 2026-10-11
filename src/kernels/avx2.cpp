// x86_64 AVX2 backend. This file is compiled with -mavx2 (see CMakeLists.txt)
// and is only ever *called* after dispatch.cpp has confirmed the CPU supports
// AVX2, so the rest of the library stays runnable on older x86 CPUs.

#if defined(FF_HAVE_AVX2) && defined(__AVX2__)

#include <immintrin.h>

#include <cstdlib>
#include <cstring>

#include "impl.hpp"

namespace ff::detail {
namespace {

inline uint64_t hsum_epi64(__m256i v) {
    const __m128i s = _mm_add_epi64(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    return static_cast<uint64_t>(_mm_cvtsi128_si64(s)) + static_cast<uint64_t>(_mm_extract_epi64(s, 1));
}

inline uint64_t hsum_epi32_u(__m256i v) {
    alignas(32) uint32_t t[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(t), v);
    uint64_t s = 0;
    for (uint32_t x : t) s += x;
    return s;
}

uint64_t sad_avx2(const uint8_t* a, ptrdiff_t as, const uint8_t* b, ptrdiff_t bs, int w, int h) {
    uint64_t total = 0;
    for (int y = 0; y < h; ++y) {
        const uint8_t* pa = a + y * as;
        const uint8_t* pb = b + y * bs;
        __m256i acc = _mm256_setzero_si256();
        int x = 0;
        for (; x + 32 <= w; x += 32) {
            // psadbw: 4 x (sum of 8 absolute differences) into u64 lanes
            acc = _mm256_add_epi64(acc, _mm256_sad_epu8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(pa + x)),
                                                        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pb + x))));
        }
        uint64_t row = hsum_epi64(acc);
        if (x + 16 <= w) {
            const __m128i s = _mm_sad_epu8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pa + x)),
                                           _mm_loadu_si128(reinterpret_cast<const __m128i*>(pb + x)));
            row += static_cast<uint64_t>(_mm_cvtsi128_si64(s)) + static_cast<uint64_t>(_mm_extract_epi64(s, 1));
            x += 16;
        }
        for (; x < w; ++x) row += static_cast<uint64_t>(std::abs(pa[x] - pb[x]));
        total += row;
    }
    return total;
}

uint64_t ssd_avx2(const uint8_t* a, ptrdiff_t as, const uint8_t* b, ptrdiff_t bs, int w, int h) {
    uint64_t total = 0;
    for (int y = 0; y < h; ++y) {
        const uint8_t* pa = a + y * as;
        const uint8_t* pb = b + y * bs;
        __m256i acc = _mm256_setzero_si256();
        int x = 0;
        for (; x + 16 <= w; x += 16) {
            const __m256i va = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pa + x)));
            const __m256i vb = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pb + x)));
            const __m256i d = _mm256_sub_epi16(va, vb);
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d, d));  // pairwise d*d + d*d into i32
        }
        uint64_t row = hsum_epi32_u(acc);
        for (; x < w; ++x) {
            const int d = pa[x] - pb[x];
            row += static_cast<uint64_t>(d * d);
        }
        total += row;
    }
    return total;
}

uint64_t sum_avx2(const uint8_t* p, ptrdiff_t stride, int w, int h) {
    uint64_t total = 0;
    const __m256i zero = _mm256_setzero_si256();
    for (int y = 0; y < h; ++y) {
        const uint8_t* r = p + y * stride;
        __m256i acc = zero;
        int x = 0;
        for (; x + 32 <= w; x += 32)
            acc = _mm256_add_epi64(acc, _mm256_sad_epu8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(r + x)), zero));
        uint64_t s = hsum_epi64(acc);
        for (; x < w; ++x) s += r[x];
        total += s;
    }
    return total;
}

void blend_rows_avx2(const uint8_t* a, const uint8_t* b, uint8_t* dst, int n, int wb) {
    if (wb == 0) {
        std::memcpy(dst, a, static_cast<size_t>(n));
        return;
    }
    const __m256i w1 = _mm256_set1_epi16(static_cast<short>(wb));
    const __m256i w0 = _mm256_set1_epi16(static_cast<short>(256 - wb));
    const __m256i round = _mm256_set1_epi16(128);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m256i va = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i)));
        const __m256i vb = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i)));
        // a*w0 + b*w1 <= 255*256 = 65280, +128 still fits an unsigned 16-bit lane
        __m256i v = _mm256_add_epi16(_mm256_mullo_epi16(va, w0), _mm256_mullo_epi16(vb, w1));
        v = _mm256_srli_epi16(_mm256_add_epi16(v, round), 8);
        // pack 16 x u16 (all <= 255) to bytes; packus works per 128-bit lane, so fix the order
        const __m256i packed = _mm256_permute4x64_epi64(_mm256_packus_epi16(v, v), 0x08);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), _mm256_castsi256_si128(packed));
    }
    const int wa = 256 - wb;
    for (; i < n; ++i) dst[i] = static_cast<uint8_t>((a[i] * wa + b[i] * wb + 128) >> 8);
}

void downscale2x_row_avx2(const uint8_t* r0, const uint8_t* r1, uint8_t* dst, int n_out) {
    const __m256i ones = _mm256_set1_epi8(1);
    const __m256i two = _mm256_set1_epi16(2);
    int i = 0;
    for (; i + 16 <= n_out; i += 16) {
        // pmaddubsw with 1s = horizontal pair sums of u8 into i16
        const __m256i s0 = _mm256_maddubs_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(r0 + 2 * i)), ones);
        const __m256i s1 = _mm256_maddubs_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(r1 + 2 * i)), ones);
        __m256i v = _mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(s0, s1), two), 2);
        const __m256i packed = _mm256_permute4x64_epi64(_mm256_packus_epi16(v, v), 0x08);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), _mm256_castsi256_si128(packed));
    }
    for (; i < n_out; ++i)
        dst[i] = static_cast<uint8_t>((r0[2 * i] + r0[2 * i + 1] + r1[2 * i] + r1[2 * i + 1] + 2) >> 2);
}

}  // namespace

const KernelTable& avx2_table() {
    static const KernelTable t{Isa::Avx2,  sad_avx2,         ssd_avx2,
                               sum_avx2,   blend_rows_avx2,  downscale2x_row_avx2};
    return t;
}

}  // namespace ff::detail

#endif  // FF_HAVE_AVX2 && __AVX2__
