#include <algorithm>
#include <cmath>
#include <random>

#include "kernels/dispatch.hpp"
#include "quality/bdrate.hpp"
#include "quality/psnr.hpp"
#include "quality/ssim.hpp"
#include "test_util.hpp"

using namespace ff;

TEST(psnr_of_identical_frames_is_capped_at_100) {
    Frame a(64, 64);
    fftest::make_moving_frame(a, 3);
    CHECK_NEAR(psnr_y(a, a), 100.0, 1e-12);
    CHECK_NEAR(psnr_yuv(a, a), 100.0, 1e-12);
}

TEST(psnr_matches_the_textbook_formula) {
    Frame a(64, 64), b(64, 64);
    a.fill(100, 128, 128);
    b.fill(101, 128, 128);  // MSE 1 on luma
    CHECK_NEAR(psnr_y(a, b), 10.0 * std::log10(255.0 * 255.0), 1e-9);  // 48.1308 dB
    // MSE 4 -> 6.0206 dB lower
    b.fill(102, 128, 128);
    CHECK_NEAR(psnr_y(a, b), 10.0 * std::log10(255.0 * 255.0 / 4.0), 1e-9);
    // Chroma untouched: the all-plane figure is higher than luma-only (error spread over 1.5x samples).
    CHECK(psnr_yuv(a, b) > psnr_y(a, b));
}

TEST(psnr_is_identical_on_every_backend) {
    Frame a(176, 144), b(176, 144);
    fftest::make_noise_frame(a, 1);
    fftest::make_noise_frame(b, 2);
    const Isa original = kernels().isa;
    std::vector<double> vals;
    for (Isa isa : supported_isas()) {
        set_active_isa(isa);
        vals.push_back(psnr_yuv(a, b));
    }
    set_active_isa(original);
    for (double v : vals) CHECK_NEAR(v, vals[0], 0.0);
}

TEST(psnr_rejects_mismatched_sizes) {
    Frame a(64, 64), b(32, 32);
    bool threw = false;
    try {
        psnr_y(a, b);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

TEST(ssim_identical_is_one_and_noise_lowers_it) {
    Frame ref(96, 64), noisy1(96, 64), noisy2(96, 64);
    fftest::make_moving_frame(ref, 0);
    noisy1.copy_from(ref);
    noisy2.copy_from(ref);
    std::mt19937 rng(5);
    for (int y = 0; y < 64; ++y)
        for (int x = 0; x < 96; ++x) {
            int n1 = int(rng() % 21) - 10, n2 = int(rng() % 81) - 40;
            noisy1.data(0)[y * noisy1.stride(0) + x] = uint8_t(std::clamp(ref.data(0)[y * ref.stride(0) + x] + n1, 0, 255));
            noisy2.data(0)[y * noisy2.stride(0) + x] = uint8_t(std::clamp(ref.data(0)[y * ref.stride(0) + x] + n2, 0, 255));
        }
    CHECK_NEAR(ssim_y(ref, ref), 1.0, 1e-12);
    const double s1 = ssim_y(ref, noisy1), s2 = ssim_y(ref, noisy2);
    CHECK(s1 < 1.0);
    CHECK(s2 < s1);  // more noise, lower similarity
    CHECK(s2 > 0.0);
}

TEST(ssim_handles_tiny_planes) {
    uint8_t a[16], b[16];
    for (int i = 0; i < 16; ++i) a[i] = b[i] = uint8_t(i * 10);
    CHECK_NEAR(ssim_plane(a, 4, b, 4, 4, 4), 1.0, 1e-12);
}

TEST(ssim_detects_structure_change_that_psnr_misses) {
    // A brightness shift keeps structure (SSIM stays high); scrambling pixels does not.
    Frame ref(64, 64), shifted(64, 64), scrambled(64, 64);
    fftest::make_moving_frame(ref, 0);
    shifted.copy_from(ref);
    scrambled.copy_from(ref);
    std::mt19937 rng(9);
    for (int y = 0; y < 64; ++y)
        for (int x = 0; x < 64; ++x) {
            const ptrdiff_t i = y * ref.stride(0) + x;
            shifted.data(0)[i] = uint8_t(std::min(255, ref.data(0)[i] + 6));
            scrambled.data(0)[i] = uint8_t(rng());
        }
    CHECK(ssim_y(ref, shifted) > 0.9);
    CHECK(ssim_y(ref, scrambled) < 0.2);
}

TEST(bd_rate_of_identical_curves_is_zero) {
    std::vector<RdPoint> c = {{1000, 34}, {2000, 37}, {4000, 40}, {8000, 42}};
    CHECK_NEAR(bd_rate(c, c), 0.0, 1e-9);
}

TEST(bd_rate_recovers_a_known_bitrate_saving) {
    std::vector<RdPoint> anchor = {{1000, 34}, {2000, 37}, {4000, 40}, {8000, 42}}, test;
    for (auto p : anchor) test.push_back({p.kbps * 0.8, p.quality});  // 20% fewer bits everywhere
    CHECK_NEAR(bd_rate(anchor, test), -20.0, 1e-6);
    for (auto& p : test) p.kbps = p.kbps / 0.8 * 1.25;  // 25% more bits
    CHECK_NEAR(bd_rate(anchor, test), 25.0, 1e-6);
}

TEST(bd_rate_uses_only_the_overlapping_quality_range) {
    std::vector<RdPoint> a = {{1000, 30}, {2000, 34}, {4000, 38}, {8000, 42}};
    std::vector<RdPoint> b = {{800, 34}, {1600, 38}, {3200, 42}, {6400, 46}};  // half-overlap, 20% cheaper at equal quality
    std::vector<RdPoint> a_shift = a;
    // Same ratio on the overlap [34, 42]: b is at 0.8x bitrate of a at the same quality only if a's curve is exponential in quality.
    // Check the weaker, robust property: b is cheaper.
    CHECK(bd_rate(a, b) < 0.0);
    // Disjoint ranges: undefined.
    std::vector<RdPoint> far = {{100, 10}, {200, 12}, {400, 14}, {800, 16}};
    CHECK(std::isnan(bd_rate(a, far)));
}

TEST(bd_rate_works_with_three_points) {
    std::vector<RdPoint> a = {{1000, 34}, {2000, 38}, {4000, 41}}, b;
    for (auto p : a) b.push_back({p.kbps * 0.5, p.quality});
    CHECK_NEAR(bd_rate(a, b), -50.0, 1e-6);
}
