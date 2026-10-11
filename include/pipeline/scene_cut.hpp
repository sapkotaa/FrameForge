#pragma once
// Scene-cut detection on downscaled luma.
//
// Each frame's luma is box-filtered down 4x (using the SIMD downscale kernel),
// then compared with the previous frame using the SIMD SAD kernel. A cut is
// declared when the mean absolute difference is both large in absolute terms
// and several times the recent average, so fast camera motion (high but
// steady difference) does not trigger it. Cuts are where chunked encoding
// prefers to split, since a scene change needs an I-frame anyway.

#include <deque>
#include <vector>

#include "pipeline/frame.hpp"

namespace ff {

struct SceneCutConfig {
    double abs_threshold = 20.0;  // mean |luma diff| on a 0..255 scale
    double ratio = 3.0;           // must also exceed ratio * recent average
    size_t history = 8;
};

class SceneCutDetector {
public:
    explicit SceneCutDetector(SceneCutConfig cfg = {}) : cfg_(cfg) {}

    // Feed frames in display order. Returns true if this frame starts a new
    // scene. The first frame never counts as a cut.
    bool push(const Frame& f);

    double last_score() const noexcept { return last_score_; }

private:
    SceneCutConfig cfg_;
    std::vector<uint8_t> prev_, cur_, tmp_;
    int pw_ = 0, ph_ = 0;
    bool have_prev_ = false;
    std::deque<double> history_;
    double last_score_ = 0;
};

std::vector<int> detect_scene_cuts(const std::vector<FramePtr>& frames, SceneCutConfig cfg = {});

}  // namespace ff
