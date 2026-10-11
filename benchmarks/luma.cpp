// SAD / SSD / plane-sum kernels: scalar vs the SIMD backends on this machine.

#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "bench_util.hpp"
#include "kernels/dispatch.hpp"

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

struct Row {
    std::string name;
    std::map<Isa, double> sec;
    double bytes;
};

void print(const std::vector<Row>& rows) {
    const auto isas = supported_isas();
    std::printf("| kernel |");
    for (Isa i : isas) std::printf(" %s GB/s |", to_string(i));
    std::printf(" speedup vs scalar |\n|---|");
    for (size_t i = 0; i < isas.size(); ++i) std::printf("---:|");
    std::printf("---:|\n");
    for (const auto& r : rows) {
        std::printf("| %s |", r.name.c_str());
        for (Isa i : isas) std::printf(" %.2f |", r.bytes / r.sec.at(i) / 1e9);
        const Isa best = isas.back();
        std::printf(" %.1fx |\n", r.sec.at(Isa::Scalar) / r.sec.at(best));
    }
}
}  // namespace

int main() {
    const int W = 1920, H = 1080;
    const auto a = noise(static_cast<size_t>(W * H), 1), b = noise(static_cast<size_t>(W * H), 2);
    std::printf("# Luma kernel benchmarks (best active ISA: %s)\n", to_string(best_isa()));
    bench::header("1080p luma plane (2 MB per operand)");

    std::vector<Row> rows;
    auto add = [&](const char* name, double bytes, auto make_fn) {
        Row r{name, {}, bytes};
        for (Isa isa : supported_isas()) r.sec[isa] = bench::seconds_per_call(make_fn(kernels_for(isa)));
        rows.push_back(std::move(r));
    };

    add("SAD 1920x1080", 2.0 * W * H, [&](const KernelTable& k) { return [&] { return k.sad(a.data(), W, b.data(), W, W, H); }; });
    add("SSD 1920x1080", 2.0 * W * H, [&](const KernelTable& k) { return [&] { return k.ssd(a.data(), W, b.data(), W, W, H); }; });
    add("plane sum 1920x1080", 1.0 * W * H, [&](const KernelTable& k) { return [&] { return k.sum(a.data(), W, W, H); }; });

    // Motion-search shaped work: many small 16x16 SADs with a strided window.
    add("SAD 16x16 blocks (motion search)", 2.0 * W * (H / 16 * 16), [&](const KernelTable& k) {
        return [&] {
            uint64_t t = 0;
            for (int by = 0; by + 16 <= H; by += 16)
                for (int bx = 0; bx + 16 <= W; bx += 16)
                    t += k.sad(a.data() + by * W + bx, W, b.data() + by * W + bx, W, 16, 16);
            return t;
        };
    });
    print(rows);

    bench::header("Effect of block width on SAD (GB/s on the best ISA vs scalar)");
    std::printf("| block | scalar GB/s | %s GB/s |\n|---|---:|---:|\n", to_string(best_isa()));
    for (int bw : {8, 16, 32, 64, 128, 1920}) {
        const int bh = 16;
        double gbs[2];
        int idx = 0;
        for (Isa isa : {Isa::Scalar, best_isa()}) {
            const auto& k = kernels_for(isa);
            const double s = bench::seconds_per_call([&] { return k.sad(a.data(), W, b.data(), W, bw, bh); });
            gbs[idx++] = 2.0 * bw * bh / s / 1e9;
        }
        std::printf("| %dx%d | %.2f | %.2f |\n", bw, bh, gbs[0], gbs[1]);
    }
    return 0;
}
