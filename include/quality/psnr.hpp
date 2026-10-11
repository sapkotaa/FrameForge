#pragma once
#include <cstddef>
#include <cstdint>

#include "pipeline/frame.hpp"

namespace ff {

// PSNR in dB for 8-bit samples. Identical inputs (MSE 0) are reported as 100 dB.
double psnr_from_mse(double mse, double peak = 255.0);

// Plane PSNR using the dispatched SSD kernel.
double psnr_plane(const uint8_t* a, ptrdiff_t as, const uint8_t* b, ptrdiff_t bs, int w, int h);

// Luma-only PSNR. This is the number used for rate-quality curves.
double psnr_y(const Frame& a, const Frame& b);

// PSNR over all samples of all planes (total squared error / total samples).
double psnr_yuv(const Frame& a, const Frame& b);

}  // namespace ff
