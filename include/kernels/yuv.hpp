#pragma once
#include <cstddef>
#include <cstdint>

namespace ff {

class Frame;

// Bytes in one 8-bit 4:2:0 frame: 1.5 samples per pixel.
inline constexpr size_t i420_frame_bytes(int w, int h) {
    return static_cast<size_t>(w) * static_cast<size_t>(h) * 3 / 2;
}

// Uncompressed bitrate in Mbps. 1080p30 8-bit 4:2:0 is ~746 Mbps, which is why
// a 5 Mbps stream is roughly 150x smaller than raw.
inline constexpr double raw_bitrate_mbps(int w, int h, double fps) {
    return static_cast<double>(i420_frame_bytes(w, h)) * 8.0 * fps / 1e6;
}

// RGB24 (R,G,B byte order) -> I420, BT.709 matrix, limited range (Y 16-235,
// Cb/Cr 16-240). Chroma is the average of each 2x2 block (4:2:0 subsampling).
void rgb24_to_i420(const uint8_t* rgb, ptrdiff_t rgb_stride, Frame& out);

}  // namespace ff





