// Resize-path kernels and the full resizer, per ISA.

#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "bench_util.hpp"
#include "kernels/dispatch.hpp"
#include "kernels/resize.hpp"
#include "pipeline/frame.hpp"

using namespace ff;

namespace {
std::vector<uint8_t> noise(size_t n, uint32_t seed) {
    std::vector<uint8_t> v(n);
    uint32_t s = seed;
    for (auto& b : v) {
        s = s * 1664525u + 1013904223u;
        b = static_cast<uint8_t>(s >> 24);
    }
    return v;
}
}  // namespace

int main() {
    const int W = 1920, H = 1080;
    const auto src = noise(static_cast<size_t>(W * H), 7);
    std::printf("# Resize benchmarks (best active ISA: %s)\n", to_string(best_isa()));
    const Isa original = kernels().isa;
    const auto isas = supported_isas();

    struct Case {
        std::string name;
        double out_mpix;
        std::map<Isa, double> sec;
    };
    std::vector<Case> cases;

    auto run = [&](const std::string& name, double out_mpix, auto fn) {
        Case c{name, out_mpix, {}};
        for (Isa isa : isas) {
            set_active_isa(isa);
            c.sec[isa] = bench::seconds_per_call(fn);
        }
        cases.push_back(std::move(c));
    };

    std::vector<uint8_t> row_a = noise(static_cast<size_t>(W), 3), row_b = noise(static_cast<size_t>(W), 4), row_o(W);
    run("blend_rows (1920 px row)", W / 1e6, [&] {
        kernels().blend_rows(row_a.data(), row_b.data(), row_o.data(), W, 77);
        return uint64_t(row_o[5]);
    });

    std::vector<uint8_t> half(static_cast<size_t>(W / 2 * (H / 2)));
    run("downscale2x 1080p -> 540p", (W / 2) * (H / 2) / 1e6, [&] {
        downscale2x(src.data(), W, W, H, half.data(), W / 2);
        return uint64_t(half[99]);
    });

    std::vector<uint8_t> out720(1280 * 720), out360(640 * 360), out240(426 * 240), up(3840 * 2160);
    run("resize_plane 1080p -> 720p (bilinear)", 1280 * 720 / 1e6, [&] {
        resize_plane(src.data(), W, W, H, out720.data(), 1280, 1280, 720);
        return uint64_t(out720[99]);
    });
    run("resize_plane 1080p -> 360p (pyramid + bilinear)", 640 * 360 / 1e6, [&] {
        resize_plane(src.data(), W, W, H, out360.data(), 640, 640, 360);
        return uint64_t(out360[99]);
    });
    run("resize_plane 1080p -> 240p (pyramid x2 + bilinear)", 426 * 240 / 1e6, [&] {
        resize_plane(src.data(), W, W, H, out240.data(), 426, 426, 240);
        return uint64_t(out240[99]);
    });
    run("resize_plane 1080p -> 2160p (upscale)", 3840 * 2160 / 1e6, [&] {
        resize_plane(src.data(), W, W, H, up.data(), 3840, 3840, 2160);
        return uint64_t(up[99]);
    });

    Frame f1080(1920, 1080), f720(1280, 720);
    f1080.fill(100, 120, 130);
    run("resize_frame I420 1080p -> 720p", 1280 * 720 * 1.5 / 1e6, [&] {
        resize_frame(f1080, f720);
        return uint64_t(f720.data(0)[5]);
    });
    set_active_isa(original);

    std::printf("\n| case |");
    for (Isa i : isas) std::printf(" %s us |", to_string(i));
    std::printf(" speedup |\n|---|");
    for (size_t i = 0; i < isas.size(); ++i) std::printf("---:|");
    std::printf("---:|\n");
    for (const auto& c : cases) {
        std::printf("| %s |", c.name.c_str());
        for (Isa i : isas) std::printf(" %.1f |", c.sec.at(i) * 1e6);
        std::printf(" %.2fx |\n", c.sec.at(Isa::Scalar) / c.sec.at(isas.back()));
    }
    std::printf(
        "\nNote: the horizontal pass of resize_plane is a gather and stays scalar, so\n"
        "end-to-end speedup is bounded by Amdahl's law; the vertical blend and the\n"
        "2x2 pyramid are SIMD.\n");
    return 0;
}
