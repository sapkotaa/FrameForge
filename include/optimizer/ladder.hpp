#pragma once
// Per-title bitrate ladder selection.
//
// Probing encodes a clip at several (resolution, CRF) pairs and measures the
// bitrate and quality of each. A point that is dominated (another probe gives
// at least as much quality for no more bits) is never worth shipping. The
// non-dominated points are the Pareto frontier; their upper convex hull is the
// idealised rate-quality curve, whose slope tells us where extra bits stop
// buying visible quality. Rungs are picked from the frontier, below that knee.

#include <limits>
#include <string>
#include <vector>

namespace ff {

struct RatePoint {
    int width = 0;
    int height = 0;
    int crf = 0;
    double kbps = 0;            // average bitrate of the encode
    double psnr = 0;            // mean luma PSNR (dB) vs the source, after upscaling to source size
    double ssim = 0;            // mean luma SSIM, same comparison
    double cpu_seconds = 0;     // time spent on the probe job (encode + decode + metrics)
};

// Non-dominated points, sorted by ascending bitrate with strictly increasing quality.
std::vector<RatePoint> pareto_frontier(std::vector<RatePoint> points);

// Upper convex hull of (log2 kbps, quality): the frontier with every point that
// lies on or below the segment between its neighbours removed. Sorted by
// ascending bitrate; slopes strictly decrease (diminishing returns).
std::vector<RatePoint> convex_hull(std::vector<RatePoint> points);

struct LadderConfig {
    int max_rungs = 6;
    double min_gap_db = 1.0;            // adjacent rungs must differ by at least this much quality
    double min_bitrate_ratio = 1.25;    // ...and by at least this bitrate ratio
    double min_db_per_doubling = 0.75;  // stop when doubling the bitrate buys less than this
    double min_kbps = 0;
    double max_kbps = std::numeric_limits<double>::infinity();
};

// Chooses the rungs from raw probe results:
//  1. Find the knee: the last hull point whose incoming slope is still at least
//     min_db_per_doubling (dB per doubling of bitrate). Nothing above it is offered.
//  2. Walk the Pareto frontier upward from the cheapest point, keeping a point
//     only if it adds min_gap_db of quality and min_bitrate_ratio of bitrate
//     over the previous rung, until max_rungs or the knee.
// The result has strictly increasing bitrate and quality.
std::vector<RatePoint> select_ladder(const std::vector<RatePoint>& points, const LadderConfig& cfg);

// Compute cost of one encode job in cost per minute of *video*:
//   cpu_seconds / video_seconds * price_per_cpu_hour / 60
double cost_per_video_minute(double cpu_seconds, double video_seconds, double price_per_cpu_hour);

// Number of views after which an extra `encode_cost_usd` is paid back by
// bandwidth savings of `saved_gb_per_view` at `usd_per_gb`. Infinity if nothing is saved.
double break_even_views(double encode_cost_usd, double saved_gb_per_view, double usd_per_gb);

}  // namespace ff
