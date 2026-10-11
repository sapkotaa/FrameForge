#include "pipeline/frame.hpp"

#include <cstring>
#include <stdexcept>

namespace ff {
namespace {
constexpr size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }
}  // namespace

Frame::Frame(int width, int height) : width_(width), height_(height) {
    if (width <= 0 || height <= 0 || (width % 2) != 0 || (height % 2) != 0)
        throw std::invalid_argument("Frame: dimensions must be positive and even");

    stride_[0] = static_cast<ptrdiff_t>(align_up(static_cast<size_t>(width), kAlignment));
    stride_[1] = stride_[2] = static_cast<ptrdiff_t>(align_up(static_cast<size_t>(width / 2), kAlignment));
    const size_t ysz = static_cast<size_t>(stride_[0]) * static_cast<size_t>(height);
    const size_t csz = static_cast<size_t>(stride_[1]) * static_cast<size_t>(height / 2);
    const size_t total = ysz + 2 * csz;  // every term is a multiple of 64

    void* mem = std::aligned_alloc(kAlignment, total);
    if (!mem) throw std::bad_alloc();
    std::memset(mem, 0, total);
    storage_.reset(static_cast<uint8_t*>(mem));
    planes_[0] = storage_.get();
    planes_[1] = planes_[0] + ysz;
    planes_[2] = planes_[1] + csz;
}

void Frame::fill(uint8_t y, uint8_t u, uint8_t v) {
    const uint8_t val[3] = {y, u, v};
    for (int p = 0; p < 3; ++p)
        for (int r = 0; r < plane_height(p); ++r)
            std::memset(planes_[p] + r * stride_[p], val[p], static_cast<size_t>(plane_width(p)));
}

void Frame::copy_from(const Frame& o) {
    if (o.width_ != width_ || o.height_ != height_) throw std::invalid_argument("Frame::copy_from: size mismatch");
    for (int p = 0; p < 3; ++p)
        for (int r = 0; r < plane_height(p); ++r)
            std::memcpy(planes_[p] + r * stride_[p], o.planes_[p] + r * o.stride_[p], static_cast<size_t>(plane_width(p)));
    pts = o.pts;
}

std::shared_ptr<FramePool> make_frame_pool(int width, int height, size_t count) {
    return FramePool::create(count, [=] { return std::make_unique<Frame>(width, height); });
}

}  // namespace ff
