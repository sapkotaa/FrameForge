#include "quality/psnr.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "kernels/luma.hpp"

namespace ff {

double psnr_from_mse(double mse, double peak) {
    if (mse <= 0.0) return 100.0;
    return std::min(100.0, 10.0 * std::log10(peak * peak / mse));
}

double psnr_plane(const uint8_t* a, ptrdiff_t as, const uint8_t* b, ptrdiff_t bs, int w, int h) {
    const double n = static_cast<double>(w) * h;
    return psnr_from_mse(static_cast<double>(ssd(a, as, b, bs, w, h)) / n);
}

static void check_same(const Frame& a, const Frame& b) {
    if (a.width() != b.width() || a.height() != b.height()) throw std::invalid_argument("psnr: frame size mismatch");
}

double psnr_y(const Frame& a, const Frame& b) {
    check_same(a, b);
    return psnr_plane(a.data(0), a.stride(0), b.data(0), b.stride(0), a.width(), a.height());
}

double psnr_yuv(const Frame& a, const Frame& b) {
    check_same(a, b);
    uint64_t err = 0;
    double samples = 0;
    for (int p = 0; p < 3; ++p) {
        err += ssd(a.data(p), a.stride(p), b.data(p), b.stride(p), a.plane_width(p), a.plane_height(p));
        samples += static_cast<double>(a.plane_width(p)) * a.plane_height(p);
    }
    return psnr_from_mse(static_cast<double>(err) / samples);
}

}  // namespace ff
