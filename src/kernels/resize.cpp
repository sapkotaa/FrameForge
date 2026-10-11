#include "kernels/resize.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

#include "kernels/dispatch.hpp"
#include "pipeline/frame.hpp"

namespace ff {

void downscale2x(const uint8_t* src, ptrdiff_t ss, int sw, int sh, uint8_t* dst, ptrdiff_t ds) {
    const auto row_fn = kernels().downscale2x_row;
    const int dw = sw / 2;
    const int dh = sh / 2;
    for (int y = 0; y < dh; ++y) row_fn(src + 2 * y * ss, src + (2 * y + 1) * ss, dst + y * ds, dw);
}

namespace {

// Center-aligned source position in 16.16 fixed point -> (index0, index1, 8-bit weight of index1).
void make_taps(int src_n, int dst_n, std::vector<int>& i0, std::vector<int>& i1, std::vector<uint8_t>& frac) {
    i0.resize(static_cast<size_t>(dst_n));
    i1.resize(static_cast<size_t>(dst_n));
    frac.resize(static_cast<size_t>(dst_n));
    const int64_t step = (static_cast<int64_t>(src_n) << 16) / dst_n;
    for (int i = 0; i < dst_n; ++i) {
        int64_t pos = static_cast<int64_t>(i) * step + (step >> 1) - 32768;
        if (pos < 0) pos = 0;
        int a = static_cast<int>(pos >> 16);
        int f = static_cast<int>((pos >> 8) & 255);
        if (a >= src_n - 1) {
            a = src_n - 1;
            f = 0;
        }
        i0[static_cast<size_t>(i)] = a;
        i1[static_cast<size_t>(i)] = std::min(a + 1, src_n - 1);
        frac[static_cast<size_t>(i)] = static_cast<uint8_t>(f);
    }
}

}  // namespace

void resize_bilinear(const uint8_t* src, ptrdiff_t ss, int sw, int sh, uint8_t* dst, ptrdiff_t ds, int dw, int dh) {
    if (sw == dw && sh == dh) {
        for (int y = 0; y < dh; ++y) std::memcpy(dst + y * ds, src + y * ss, static_cast<size_t>(dw));
        return;
    }
    std::vector<int> x0, x1, y0, y1;
    std::vector<uint8_t> fx, fy;
    make_taps(sw, dw, x0, x1, fx);
    make_taps(sh, dh, y0, y1, fy);

    // Vertical first, horizontal second. The vertical blend runs full-width on the
    // SIMD kernel and leaves one row per output row; the horizontal pass (a gather,
    // which stays scalar) then only touches dw samples instead of 2 * dw. Profiling
    // the other order showed that scalar pass was ~80% of the runtime.
    const auto blend = kernels().blend_rows;
    std::vector<uint8_t> tmp(static_cast<size_t>(sw));
    for (int y = 0; y < dh; ++y) {
        const size_t yi = static_cast<size_t>(y);
        blend(src + y0[yi] * ss, src + y1[yi] * ss, tmp.data(), sw, fy[yi]);
        uint8_t* out = dst + y * ds;
        for (int x = 0; x < dw; ++x) {
            const size_t xi = static_cast<size_t>(x);
            const int f = fx[xi];
            out[x] = static_cast<uint8_t>((tmp[static_cast<size_t>(x0[xi])] * (256 - f) + tmp[static_cast<size_t>(x1[xi])] * f + 128) >> 8);
        }
    }
}

void resize_plane(const uint8_t* src, ptrdiff_t ss, int sw, int sh, uint8_t* dst, ptrdiff_t ds, int dw, int dh) {
    std::vector<uint8_t> buf[2];
    int which = 0;
    const uint8_t* cur = src;
    ptrdiff_t cs = ss;
    int cw = sw, ch = sh;
    while (dw * 2 <= cw && dh * 2 <= ch) {
        const int nw = cw / 2, nh = ch / 2;
        buf[which].resize(static_cast<size_t>(nw) * static_cast<size_t>(nh));
        downscale2x(cur, cs, cw, ch, buf[which].data(), nw);
        cur = buf[which].data();
        cs = nw;
        cw = nw;
        ch = nh;
        which ^= 1;
    }
    resize_bilinear(cur, cs, cw, ch, dst, ds, dw, dh);
}

void resize_frame(const Frame& src, Frame& dst) {
    for (int p = 0; p < 3; ++p)
        resize_plane(src.data(p), src.stride(p), src.plane_width(p), src.plane_height(p), dst.data(p), dst.stride(p),
                     dst.plane_width(p), dst.plane_height(p));
    dst.pts = src.pts;
}

}  // namespace ff
