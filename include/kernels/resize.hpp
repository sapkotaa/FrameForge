#pragma once
#include <cstddef>
#include <cstdint>

namespace ff {

class Frame;

// 2x2 box downscale. Output is (sw/2) x (sh/2); an odd trailing row/column is dropped.
void downscale2x(const uint8_t* src, ptrdiff_t src_stride, int sw, int sh,
                 uint8_t* dst, ptrdiff_t dst_stride);

// Fixed-point (8-bit weights) bilinear resize with center-aligned sampling.
// Only suitable for modest ratios; resize_plane() adds the pyramid.
void resize_bilinear(const uint8_t* src, ptrdiff_t src_stride, int sw, int sh,
                     uint8_t* dst, ptrdiff_t dst_stride, int dw, int dh);

// Box-downscale by 2x while the target is at most half the size (cheap
// anti-aliasing), then one bilinear step to the exact size.
void resize_plane(const uint8_t* src, ptrdiff_t src_stride, int sw, int sh,
                  uint8_t* dst, ptrdiff_t dst_stride, int dw, int dh);

// Resize all three planes of an I420 frame. dst dimensions come from dst itself.
void resize_frame(const Frame& src, Frame& dst);

}  // namespace ff
