#pragma once
// Plane-level helpers over the dispatched kernels: block SAD/SSD (motion
// search cost, PSNR) and luma energy (brightness, scene-cut analysis).

#include "kernels/dispatch.hpp"

namespace ff {

inline uint64_t sad(const uint8_t* a, ptrdiff_t as, const uint8_t* b, ptrdiff_t bs, int w, int h) {
    return kernels().sad(a, as, b, bs, w, h);
}

inline uint64_t ssd(const uint8_t* a, ptrdiff_t as, const uint8_t* b, ptrdiff_t bs, int w, int h) {
    return kernels().ssd(a, as, b, bs, w, h);
}

inline uint64_t luma_sum(const uint8_t* p, ptrdiff_t stride, int w, int h) {
    return kernels().sum(p, stride, w, h);
}

inline double luma_mean(const uint8_t* p, ptrdiff_t stride, int w, int h) {
    if (w <= 0 || h <= 0) return 0.0;
    return static_cast<double>(luma_sum(p, stride, w, h)) / (static_cast<double>(w) * h);
}

}  // namespace ff
