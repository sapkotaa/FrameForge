#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>

#include "pipeline/buffer_pool.hpp"

namespace ff {

// 8-bit planar YCbCr 4:2:0 (I420). Planes are 64-byte aligned and every row
// starts on a 64-byte boundary, so SIMD loads never straddle cache lines at
// row starts. Width and height must be even.
class Frame {
public:
    static constexpr int kAlignment = 64;

    Frame(int width, int height);
    Frame(const Frame&) = delete;
    Frame& operator=(const Frame&) = delete;

    int width() const noexcept { return width_; }
    int height() const noexcept { return height_; }
    int plane_width(int p) const noexcept { return p == 0 ? width_ : width_ / 2; }
    int plane_height(int p) const noexcept { return p == 0 ? height_ : height_ / 2; }
    ptrdiff_t stride(int p) const noexcept { return stride_[p]; }
    uint8_t* data(int p) noexcept { return planes_[p]; }
    const uint8_t* data(int p) const noexcept { return planes_[p]; }

    size_t byte_size() const noexcept { return static_cast<size_t>(width_) * static_cast<size_t>(height_) * 3 / 2; }

    void fill(uint8_t y, uint8_t u, uint8_t v);
    void copy_from(const Frame& other);  // same dimensions required

    int64_t pts = 0;  // presentation index (frames) or timestamp, set by the producer

private:
    struct FreeDeleter {
        void operator()(uint8_t* p) const noexcept { std::free(p); }
    };
    int width_;
    int height_;
    ptrdiff_t stride_[3];
    uint8_t* planes_[3];
    std::unique_ptr<uint8_t, FreeDeleter> storage_;
};

using FramePtr = std::shared_ptr<Frame>;
using FramePool = ObjectPool<Frame>;

std::shared_ptr<FramePool> make_frame_pool(int width, int height, size_t count);

}  // namespace ff
