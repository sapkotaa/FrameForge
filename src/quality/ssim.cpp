#include "quality/ssim.hpp"

#include <stdexcept>
#include <vector>

namespace ff {
namespace {

struct Sums {
    uint32_t a = 0, b = 0, aa = 0, bb = 0, ab = 0;  // 4x4 block: aa <= 16 * 255^2, fits u32
};

inline double window_ssim(double sa, double sb, double saa, double sbb, double sab, double n) {
    constexpr double c1 = (0.01 * 255) * (0.01 * 255);
    constexpr double c2 = (0.03 * 255) * (0.03 * 255);
    const double ma = sa / n, mb = sb / n;
    const double va = saa / n - ma * ma;
    const double vb = sbb / n - mb * mb;
    const double cov = sab / n - ma * mb;
    return ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2));
}

}  // namespace

double ssim_plane(const uint8_t* a, ptrdiff_t as, const uint8_t* b, ptrdiff_t bs, int w, int h) {
    if (w < 8 || h < 8) {
        double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const double pa = a[y * as + x], pb = b[y * bs + x];
                sa += pa;
                sb += pb;
                saa += pa * pa;
                sbb += pb * pb;
                sab += pa * pb;
            }
        return window_ssim(sa, sb, saa, sbb, sab, static_cast<double>(w) * h);
    }

    const int bw = w / 4, bh = h / 4;
    std::vector<Sums> blocks(static_cast<size_t>(bw) * static_cast<size_t>(bh));
    for (int by = 0; by < bh; ++by) {
        for (int bx = 0; bx < bw; ++bx) {
            Sums s;
            for (int y = 0; y < 4; ++y) {
                const uint8_t* pa = a + (by * 4 + y) * as + bx * 4;
                const uint8_t* pb = b + (by * 4 + y) * bs + bx * 4;
                for (int x = 0; x < 4; ++x) {
                    const uint32_t va = pa[x], vb = pb[x];
                    s.a += va;
                    s.b += vb;
                    s.aa += va * va;
                    s.bb += vb * vb;
                    s.ab += va * vb;
                }
            }
            blocks[static_cast<size_t>(by) * static_cast<size_t>(bw) + static_cast<size_t>(bx)] = s;
        }
    }

    double total = 0;
    size_t count = 0;
    for (int by = 0; by + 1 < bh; ++by) {
        for (int bx = 0; bx + 1 < bw; ++bx) {
            uint64_t sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx) {
                    const Sums& s = blocks[static_cast<size_t>(by + dy) * static_cast<size_t>(bw) + static_cast<size_t>(bx + dx)];
                    sa += s.a;
                    sb += s.b;
                    saa += s.aa;
                    sbb += s.bb;
                    sab += s.ab;
                }
            total += window_ssim(static_cast<double>(sa), static_cast<double>(sb), static_cast<double>(saa),
                                 static_cast<double>(sbb), static_cast<double>(sab), 64.0);
            ++count;
        }
    }
    return total / static_cast<double>(count);
}

double ssim_y(const Frame& a, const Frame& b) {
    if (a.width() != b.width() || a.height() != b.height()) throw std::invalid_argument("ssim: frame size mismatch");
    return ssim_plane(a.data(0), a.stride(0), b.data(0), b.stride(0), a.width(), a.height());
}

}  // namespace ff
