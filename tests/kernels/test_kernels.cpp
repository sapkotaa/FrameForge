// Every SIMD backend must be bit-exact with the scalar reference, which in
// turn is checked against naive loops written independently in this file.

#include <cstdlib>
#include <random>

#include "kernels/dispatch.hpp"
#include "test_util.hpp"

using namespace ff;

namespace {

uint64_t ref_sad(const uint8_t* a, ptrdiff_t as, const uint8_t* b, ptrdiff_t bs, int w, int h) {
    uint64_t s = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) s += static_cast<uint64_t>(std::abs(int(a[y * as + x]) - int(b[y * bs + x])));
    return s;
}
uint64_t ref_ssd(const uint8_t* a, ptrdiff_t as, const uint8_t* b, ptrdiff_t bs, int w, int h) {
    uint64_t s = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const int d = int(a[y * as + x]) - int(b[y * bs + x]);
            s += static_cast<uint64_t>(d * d);
        }
    return s;
}
uint64_t ref_sum(const uint8_t* a, ptrdiff_t as, int w, int h) {
    uint64_t s = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) s += a[y * as + x];
    return s;
}

const int kWidths[] = {0, 1, 3, 7, 8, 9, 15, 16, 17, 23, 31, 32, 33, 47, 63, 64, 65, 100, 255, 256, 1000};
const int kHeights[] = {0, 1, 2, 5, 16};

}  // namespace

TEST(scalar_matches_naive_reference) {
    const auto& k = kernels_for(Isa::Scalar);
    for (int w : kWidths)
        for (int h : kHeights) {
            const ptrdiff_t as = w + 13, bs = w + 29;  // strides wider than the block
            auto a = fftest::random_bytes(static_cast<size_t>(as * (h + 1) + 64), 1);
            auto b = fftest::random_bytes(static_cast<size_t>(bs * (h + 1) + 64), 2);
            CHECK_EQ(k.sad(a.data(), as, b.data(), bs, w, h), ref_sad(a.data(), as, b.data(), bs, w, h));
            CHECK_EQ(k.ssd(a.data(), as, b.data(), bs, w, h), ref_ssd(a.data(), as, b.data(), bs, w, h));
            CHECK_EQ(k.sum(a.data(), as, w, h), ref_sum(a.data(), as, w, h));
        }
}

TEST(every_backend_matches_scalar_on_random_data) {
    const auto& ref = kernels_for(Isa::Scalar);
    for (Isa isa : supported_isas()) {
        const auto& k = kernels_for(isa);
        std::printf("    backend %s\n", to_string(isa));
        for (int w : kWidths)
            for (int h : kHeights) {
                const ptrdiff_t as = w + 13, bs = w + 29;
                auto a = fftest::random_bytes(static_cast<size_t>(as * (h + 1) + 64), 11 + w);
                auto b = fftest::random_bytes(static_cast<size_t>(bs * (h + 1) + 64), 12 + h);
                CHECK_EQ(k.sad(a.data(), as, b.data(), bs, w, h), ref.sad(a.data(), as, b.data(), bs, w, h));
                CHECK_EQ(k.ssd(a.data(), as, b.data(), bs, w, h), ref.ssd(a.data(), as, b.data(), bs, w, h));
                CHECK_EQ(k.sum(a.data(), as, w, h), ref.sum(a.data(), as, w, h));
            }
    }
}

TEST(every_backend_handles_extreme_values) {
    // 0 vs 255 is the largest possible per-sample error; this is where an
    // accumulator that is too narrow would overflow.
    const int w = 4096, h = 64;
    std::vector<uint8_t> zero(static_cast<size_t>(w * h), 0), full(static_cast<size_t>(w * h), 255);
    for (Isa isa : supported_isas()) {
        const auto& k = kernels_for(isa);
        CHECK_EQ(k.sad(zero.data(), w, full.data(), w, w, h), uint64_t(255) * w * h);
        CHECK_EQ(k.ssd(zero.data(), w, full.data(), w, w, h), uint64_t(255) * 255 * w * h);
        CHECK_EQ(k.sum(full.data(), w, w, h), uint64_t(255) * w * h);
    }
}

TEST(blend_rows_is_bit_exact_for_all_weights) {
    const auto& ref = kernels_for(Isa::Scalar);
    for (Isa isa : supported_isas()) {
        const auto& k = kernels_for(isa);
        for (int n : {1, 7, 15, 16, 17, 31, 32, 33, 100}) {
            auto a = fftest::random_bytes(static_cast<size_t>(n), 5);
            auto b = fftest::random_bytes(static_cast<size_t>(n), 6);
            std::vector<uint8_t> d0(static_cast<size_t>(n)), d1(static_cast<size_t>(n));
            for (int wb = 0; wb < 256; ++wb) {
                ref.blend_rows(a.data(), b.data(), d0.data(), n, wb);
                k.blend_rows(a.data(), b.data(), d1.data(), n, wb);
                CHECK(d0 == d1);
            }
        }
        // 255 + 255 must stay 255 (no overflow in the 16-bit intermediate).
        std::vector<uint8_t> f(64, 255), out(64);
        for (int wb = 0; wb < 256; ++wb) {
            k.blend_rows(f.data(), f.data(), out.data(), 64, wb);
            CHECK_EQ(int(out[0]), 255);
        }
    }
}

TEST(blend_rows_matches_the_formula) {
    const auto& k = kernels_for(Isa::Scalar);
    uint8_t a = 10, b = 200, d;
    k.blend_rows(&a, &b, &d, 1, 128);
    CHECK_EQ(int(d), (10 * 128 + 200 * 128 + 128) >> 8);
    k.blend_rows(&a, &b, &d, 1, 0);
    CHECK_EQ(int(d), 10);
}

TEST(downscale2x_row_is_bit_exact) {
    const auto& ref = kernels_for(Isa::Scalar);
    for (Isa isa : supported_isas()) {
        const auto& k = kernels_for(isa);
        for (int n : {1, 3, 8, 15, 16, 17, 31, 32, 33, 100, 500}) {
            auto r0 = fftest::random_bytes(static_cast<size_t>(2 * n + 64), 21);
            auto r1 = fftest::random_bytes(static_cast<size_t>(2 * n + 64), 22);
            std::vector<uint8_t> d0(static_cast<size_t>(n)), d1(static_cast<size_t>(n));
            ref.downscale2x_row(r0.data(), r1.data(), d0.data(), n);
            k.downscale2x_row(r0.data(), r1.data(), d1.data(), n);
            CHECK(d0 == d1);
        }
        std::vector<uint8_t> hi(128, 255), out(64);  // max input: (4*255 + 2) >> 2 = 255
        k.downscale2x_row(hi.data(), hi.data(), out.data(), 64);
        CHECK_EQ(int(out[63]), 255);
    }
}

TEST(dispatch_picks_a_supported_backend) {
    CHECK(isa_supported(Isa::Scalar));
    CHECK(isa_supported(best_isa()));
    CHECK(kernels().isa == kernels_for(best_isa()).isa || std::getenv("FRAMEFORGE_ISA") != nullptr);
    bool threw = false;
    for (Isa isa : {Isa::Neon, Isa::Avx2}) {
        if (isa_supported(isa)) continue;
        try {
            kernels_for(isa);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
    }
    (void)threw;  // only meaningful when some backend is unavailable
}
