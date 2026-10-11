#pragma once
#include <cstddef>
#include <cstdint>

#include "pipeline/frame.hpp"

namespace ff {

// Mean SSIM over 8x8 windows sampled every 4 pixels (uniform weights).
// Window statistics are assembled from 4x4 block sums, so every pixel is read
// once. Planes smaller than 8x8 are treated as a single window.
double ssim_plane(const uint8_t* a, ptrdiff_t as, const uint8_t* b, ptrdiff_t bs, int w, int h);

double ssim_y(const Frame& a, const Frame& b);

}  // namespace ff
