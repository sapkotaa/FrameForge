#include <cstdlib>
#include <tuple>
#include <utility>

#include "kernels/dispatch.hpp"
#include "kernels/luma.hpp"
#include "kernels/resize.hpp"
#include "kernels/yuv.hpp"
#include "test_util.hpp"

using namespace ff;

namespace {
struct Plane {
    int w, h;
    std::vector<uint8_t> d;
    Plane(int w_, int h_, uint32_t seed) : w(w_), h(h_), d(fftest::random_bytes(static_cast<size_t>(w_ * h_), seed)) {}
};
}  // namespace

TEST(resize_same_size_is_a_copy) {
    Plane p(37, 21, 1);
    std::vector<uint8_t> out(p.d.size());
    resize_bilinear(p.d.data(), p.w, p.w, p.h, out.data(), p.w, p.w, p.h);
    CHECK(out == p.d);
}

TEST(resize_preserves_constant_images) {
    for (auto [sw, sh, dw, dh] : {std::tuple{64, 48, 31, 17}, {16, 16, 100, 90}, {200, 100, 20, 10}, {33, 33, 33, 66}}) {
        std::vector<uint8_t> src(static_cast<size_t>(sw * sh), 137), dst(static_cast<size_t>(dw * dh), 0);
        resize_plane(src.data(), sw, sw, sh, dst.data(), dw, dw, dh);
        for (uint8_t v : dst) CHECK_EQ(int(v), 137);
    }
}

TEST(upscaled_gradient_stays_monotonic) {
    const int sw = 32, sh = 8;
    std::vector<uint8_t> src(static_cast<size_t>(sw * sh));
    for (int y = 0; y < sh; ++y)
        for (int x = 0; x < sw; ++x) src[static_cast<size_t>(y * sw + x)] = static_cast<uint8_t>(x * 8);
    const int dw = 96, dh = 24;
    std::vector<uint8_t> dst(static_cast<size_t>(dw * dh));
    resize_bilinear(src.data(), sw, sw, sh, dst.data(), dw, dw, dh);
    for (int y = 0; y < dh; ++y)
        for (int x = 1; x < dw; ++x) CHECK(dst[static_cast<size_t>(y * dw + x)] >= dst[static_cast<size_t>(y * dw + x - 1)]);
}

TEST(downscale2x_averages_blocks) {
    // 4x2 input -> 2x1 output
    const uint8_t src[8] = {10, 20, 100, 110, 30, 40, 120, 130};
    uint8_t dst[2];
    downscale2x(src, 4, 4, 2, dst, 2);
    CHECK_EQ(int(dst[0]), (10 + 20 + 30 + 40 + 2) / 4);
    CHECK_EQ(int(dst[1]), (100 + 110 + 120 + 130 + 2) / 4);
}

TEST(resize_result_is_identical_on_every_backend) {
    Plane p(203, 117, 9);
    const Isa original = kernels().isa;
    std::vector<std::vector<uint8_t>> outs;
    for (Isa isa : supported_isas()) {
        set_active_isa(isa);
        for (auto [dw, dh] : {std::pair{50, 30}, {400, 250}, {101, 59}}) {
            std::vector<uint8_t> out(static_cast<size_t>(dw * dh));
            resize_plane(p.d.data(), p.w, p.w, p.h, out.data(), dw, dw, dh);
            outs.push_back(std::move(out));
        }
    }
    set_active_isa(original);
    const size_t per_isa = 3;
    for (size_t i = per_isa; i < outs.size(); ++i) CHECK(outs[i] == outs[i % per_isa]);
}

TEST(frame_resize_handles_all_planes) {
    Frame a(64, 48), b(32, 24), c(128, 96);
    a.fill(100, 90, 160);
    resize_frame(a, b);
    resize_frame(a, c);
    for (Frame* f : {&b, &c}) {
        CHECK_EQ(int(f->data(0)[0]), 100);
        CHECK_EQ(int(f->data(1)[0]), 90);
        CHECK_EQ(int(f->data(2)[f->plane_width(2) - 1 + (f->plane_height(2) - 1) * f->stride(2)]), 160);
    }
}

TEST(luma_mean_matches_fill) {
    Frame f(64, 32);
    f.fill(77, 128, 128);
    CHECK_NEAR(luma_mean(f.data(0), f.stride(0), 64, 32), 77.0, 1e-9);
    CHECK_EQ(luma_sum(f.data(0), f.stride(0), 64, 32), uint64_t(77) * 64 * 32);
}

TEST(raw_bitrate_matches_the_notes) {
    // 1080p30 8-bit 4:2:0 is ~93 MB/s, ~746 Mbps.
    CHECK_EQ(i420_frame_bytes(1920, 1080), size_t(3110400));
    CHECK_NEAR(raw_bitrate_mbps(1920, 1080, 30), 746.496, 0.001);
}

TEST(rgb_to_i420_hits_the_limited_range_corners) {
    std::vector<uint8_t> white(2 * 2 * 3, 255), black(2 * 2 * 3, 0), grey(2 * 2 * 3, 128);
    Frame f(2, 2);
    rgb24_to_i420(white.data(), 6, f);
    CHECK_EQ(int(f.data(0)[0]), 235);
    CHECK(std::abs(int(f.data(1)[0]) - 128) <= 1);
    rgb24_to_i420(black.data(), 6, f);
    CHECK_EQ(int(f.data(0)[0]), 16);
    rgb24_to_i420(grey.data(), 6, f);
    CHECK(std::abs(int(f.data(0)[0]) - 126) <= 1);  // 16 + 0.8588 * 128
    CHECK(std::abs(int(f.data(2)[0]) - 128) <= 1);
}
