#pragma once
#include <vector>

namespace ff {

struct RdPoint {
    double kbps;
    double quality;  // PSNR (dB) or any monotonic quality score
};

// Bjontegaard delta-rate: the average bitrate difference, in percent, of
// `test` relative to `anchor` over their overlapping quality range. Negative
// means `test` needs fewer bits for the same quality. Fits log10(rate) as a
// cubic polynomial of quality (lower degree if fewer than 4 points) and
// integrates the gap. Returns NaN if the quality ranges do not overlap.
double bd_rate(const std::vector<RdPoint>& anchor, const std::vector<RdPoint>& test);

}  // namespace ff
