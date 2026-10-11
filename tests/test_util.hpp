#pragma once
// Tiny dependency-free test framework + synthetic video helpers.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "pipeline/frame.hpp"

namespace fftest {

struct Case {
    const char* name;
    void (*fn)();
};
inline std::vector<Case>& registry() {
    static std::vector<Case> r;
    return r;
}
struct Registrar {
    Registrar(const char* n, void (*f)()) { registry().push_back({n, f}); }
};
inline int& failures() {
    static int f = 0;
    return f;
}
struct Abort {};

}  // namespace fftest

#define TEST(name)                                          \
    static void name();                                     \
    static ::fftest::Registrar reg_##name(#name, name);     \
    static void name()

#define CHECK(cond)                                                                      \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::fprintf(stderr, "    %s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ++::fftest::failures();                                                      \
        }                                                                                \
    } while (0)

#define REQUIRE(cond)                                                                    \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::fprintf(stderr, "    %s:%d: REQUIRE failed: %s\n", __FILE__, __LINE__, #cond); \
            ++::fftest::failures();                                                      \
            throw ::fftest::Abort{};                                                     \
        }                                                                                \
    } while (0)

#define CHECK_EQ(a, b)                                                                              \
    do {                                                                                            \
        const auto va_ = (a);                                                                       \
        const auto vb_ = (b);                                                                       \
        if (!(va_ == vb_)) {                                                                        \
            std::fprintf(stderr, "    %s:%d: CHECK_EQ failed: %s == %s  (%lld vs %lld)\n", __FILE__, __LINE__, #a, #b, \
                         static_cast<long long>(va_), static_cast<long long>(vb_));                 \
            ++::fftest::failures();                                                                 \
        }                                                                                           \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                                                       \
    do {                                                                                            \
        const double va_ = (a);                                                                     \
        const double vb_ = (b);                                                                     \
        if (!(std::fabs(va_ - vb_) <= (eps))) {                                                     \
            std::fprintf(stderr, "    %s:%d: CHECK_NEAR failed: %s ~ %s  (%.6f vs %.6f)\n", __FILE__, __LINE__, #a, #b, \
                         va_, vb_);                                                                 \
            ++::fftest::failures();                                                                 \
        }                                                                                           \
    } while (0)

namespace fftest {

inline std::vector<uint8_t> random_bytes(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<uint8_t> v(n);
    for (auto& b : v) b = static_cast<uint8_t>(rng());
    return v;
}

// Smooth moving gradient + a bright square travelling diagonally + mild noise.
// Compressible but not trivial, and frames differ from each other.
inline void make_moving_frame(ff::Frame& f, int t, uint32_t seed = 1) {
    std::mt19937 rng(seed + static_cast<uint32_t>(t) * 7919u);
    const int w = f.width(), h = f.height();
    for (int y = 0; y < h; ++y) {
        uint8_t* row = f.data(0) + y * f.stride(0);
        for (int x = 0; x < w; ++x) {
            int v = 40 + ((x * 3 + y * 2 + t * 4) % 160);
            const int sx = (t * 3) % std::max(1, w - 24), sy = (t * 2) % std::max(1, h - 24);
            if (x >= sx && x < sx + 24 && y >= sy && y < sy + 24) v = 235;
            v += static_cast<int>(rng() % 5) - 2;
            row[x] = static_cast<uint8_t>(std::min(255, std::max(0, v)));
        }
    }
    for (int p = 1; p < 3; ++p)
        for (int y = 0; y < f.plane_height(p); ++y) {
            uint8_t* row = f.data(p) + y * f.stride(p);
            for (int x = 0; x < f.plane_width(p); ++x) row[x] = static_cast<uint8_t>(110 + ((x + y + t) % 40) * (p == 1 ? 1 : 2));
        }
    f.pts = t;
}

inline void make_noise_frame(ff::Frame& f, uint32_t seed) {
    std::mt19937 rng(seed);
    for (int p = 0; p < 3; ++p)
        for (int y = 0; y < f.plane_height(p); ++y) {
            uint8_t* row = f.data(p) + y * f.stride(p);
            for (int x = 0; x < f.plane_width(p); ++x) row[x] = static_cast<uint8_t>(rng());
        }
}

inline std::vector<ff::FramePtr> make_clip(int w, int h, int n) {
    std::vector<ff::FramePtr> v;
    for (int i = 0; i < n; ++i) {
        auto f = std::make_shared<ff::Frame>(w, h);
        make_moving_frame(*f, i);
        v.push_back(std::move(f));
    }
    return v;
}

}  // namespace fftest
