#include "kernels/yuv.hpp"

#include <algorithm>
#include <cmath>

#include "pipeline/frame.hpp"

namespace ff {
namespace {
uint8_t clamp8(double v) { return static_cast<uint8_t>(std::clamp(std::lround(v), 0L, 255L)); }
}  // namespace

void rgb24_to_i420(const uint8_t* rgb, ptrdiff_t stride, Frame& out) {
    // BT.709, limited range: Y in [16,235], Cb/Cr in [16,240].
    constexpr double ky_r = 0.1826, ky_g = 0.6142, ky_b = 0.0620;
    constexpr double kcb_r = -0.1006, kcb_g = -0.3386, kcb_b = 0.4392;
    constexpr double kcr_r = 0.4392, kcr_g = -0.3989, kcr_b = -0.0403;

    const int w = out.width(), h = out.height();
    for (int y = 0; y < h; ++y) {
        const uint8_t* px = rgb + y * stride;
        uint8_t* yp = out.data(0) + y * out.stride(0);
        for (int x = 0; x < w; ++x)
            yp[x] = clamp8(16.0 + ky_r * px[3 * x] + ky_g * px[3 * x + 1] + ky_b * px[3 * x + 2]);
    }
    for (int cy = 0; cy < h / 2; ++cy) {
        uint8_t* up = out.data(1) + cy * out.stride(1);
        uint8_t* vp = out.data(2) + cy * out.stride(2);
        for (int cx = 0; cx < w / 2; ++cx) {
            double r = 0, g = 0, b = 0;
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx) {
                    const uint8_t* px = rgb + (2 * cy + dy) * stride + 3 * (2 * cx + dx);
                    r += px[0];
                    g += px[1];
                    b += px[2];
                }
            r /= 4;
            g /= 4;
            b /= 4;
            up[cx] = clamp8(128.0 + kcb_r * r + kcb_g * g + kcb_b * b);
            vp[cx] = clamp8(128.0 + kcr_r * r + kcr_g * g + kcr_b * b);
        }
    }
}

}  // namespace ff
