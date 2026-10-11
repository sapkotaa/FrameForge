#include "pipeline/scene_cut.hpp"

#include <algorithm>
#include <numeric>

#include "kernels/luma.hpp"
#include "kernels/resize.hpp"

namespace ff {

bool SceneCutDetector::push(const Frame& f) {
    // Luma at 1/4 size (two 2x2 box passes), unless the frame is already tiny.
    int w = f.width(), h = f.height();
    const uint8_t* src = f.data(0);
    ptrdiff_t ss = f.stride(0);
    std::vector<uint8_t>* bufs[2] = {&cur_, &tmp_};
    int which = 0;
    for (int pass = 0; pass < 2 && w >= 32 && h >= 32; ++pass) {
        const int nw = w / 2, nh = h / 2;
        bufs[which]->resize(static_cast<size_t>(nw) * static_cast<size_t>(nh));
        downscale2x(src, ss, w, h, bufs[which]->data(), nw);
        src = bufs[which]->data();
        ss = nw;
        w = nw;
        h = nh;
        which ^= 1;
    }
    // Normalise into cur_ as a tightly-packed plane.
    std::vector<uint8_t> plane(static_cast<size_t>(w) * static_cast<size_t>(h));
    for (int y = 0; y < h; ++y) std::copy_n(src + y * ss, w, plane.data() + static_cast<size_t>(y) * static_cast<size_t>(w));

    bool cut = false;
    if (have_prev_ && w == pw_ && h == ph_) {
        const uint64_t d = sad(prev_.data(), w, plane.data(), w, w, h);
        last_score_ = static_cast<double>(d) / (static_cast<double>(w) * h);
        const double avg = history_.empty()
                               ? 0.0
                               : std::accumulate(history_.begin(), history_.end(), 0.0) / static_cast<double>(history_.size());
        cut = last_score_ > cfg_.abs_threshold && last_score_ > cfg_.ratio * std::max(avg, 1.0);
        if (!cut) {
            history_.push_back(last_score_);
            if (history_.size() > cfg_.history) history_.pop_front();
        } else {
            history_.clear();  // the new scene sets its own baseline
        }
    }
    prev_ = std::move(plane);
    pw_ = w;
    ph_ = h;
    have_prev_ = true;
    return cut;
}

std::vector<int> detect_scene_cuts(const std::vector<FramePtr>& frames, SceneCutConfig cfg) {
    SceneCutDetector det(cfg);
    std::vector<int> cuts;
    for (size_t i = 0; i < frames.size(); ++i)
        if (det.push(*frames[i])) cuts.push_back(static_cast<int>(i));
    return cuts;
}

}  // namespace ff
