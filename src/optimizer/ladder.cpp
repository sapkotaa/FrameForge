#include "optimizer/ladder.hpp"

#include <algorithm>
#include <cmath>

namespace ff {
namespace {

double x_of(const RatePoint& p) { return std::log2(std::max(p.kbps, 1e-9)); }

// > 0 for a counter-clockwise turn a -> b -> c in (log2 kbps, quality) space.
double cross(const RatePoint& a, const RatePoint& b, const RatePoint& c) {
    return (x_of(b) - x_of(a)) * (c.psnr - a.psnr) - (b.psnr - a.psnr) * (x_of(c) - x_of(a));
}

}  // namespace

std::vector<RatePoint> pareto_frontier(std::vector<RatePoint> points) {
    std::sort(points.begin(), points.end(), [](const RatePoint& a, const RatePoint& b) {
        if (a.kbps != b.kbps) return a.kbps < b.kbps;
        return a.psnr > b.psnr;
    });
    // Anything that costs more but is not better than a cheaper point is dominated.
    std::vector<RatePoint> frontier;
    for (const auto& p : points) {
        if (frontier.empty() || p.psnr > frontier.back().psnr) frontier.push_back(p);
    }
    return frontier;
}

std::vector<RatePoint> convex_hull(std::vector<RatePoint> points) {
    // Upper hull (monotone chain): pop while the middle point lies on or below
    // the segment joining its neighbours.
    std::vector<RatePoint> hull;
    for (const auto& p : pareto_frontier(std::move(points))) {
        while (hull.size() >= 2 && cross(hull[hull.size() - 2], hull.back(), p) >= 0) hull.pop_back();
        hull.push_back(p);
    }
    return hull;
}

std::vector<RatePoint> select_ladder(const std::vector<RatePoint>& points, const LadderConfig& cfg) {
    const std::vector<RatePoint> hull = convex_hull(points);
    if (hull.empty()) return {};

    // Knee of the curve. The hull is concave, so slopes only fall: stop at the first weak one.
    double knee_kbps = hull.front().kbps;
    for (size_t i = 1; i < hull.size(); ++i) {
        const double slope = (hull[i].psnr - hull[i - 1].psnr) / (x_of(hull[i]) - x_of(hull[i - 1]));
        if (slope < cfg.min_db_per_doubling) break;
        knee_kbps = hull[i].kbps;
    }
    const double ceiling = std::min(knee_kbps, cfg.max_kbps);

    // Greedy pass: every candidate that is a visible step up from the previous one.
    std::vector<RatePoint> steps;
    for (const auto& p : pareto_frontier(points)) {
        if (p.kbps < cfg.min_kbps || p.kbps > ceiling) continue;
        if (steps.empty()) {
            steps.push_back(p);
            continue;
        }
        const RatePoint& last = steps.back();
        if (p.psnr - last.psnr >= cfg.min_gap_db && p.kbps / last.kbps >= cfg.min_bitrate_ratio) steps.push_back(p);
    }
    const size_t cap = static_cast<size_t>(std::max(1, cfg.max_rungs));
    if (steps.size() <= cap) return steps;

    // Too many steps: keep the cheapest and the knee, and spread the rest evenly in
    // log-bitrate (the axis viewers' bandwidth is spread over). Thinning only widens gaps,
    // so the min_gap / min_ratio guarantees still hold.
    if (cap == 1) return {steps.back()};
    const double lo = x_of(steps.front()), hi = x_of(steps.back());
    std::vector<bool> taken(steps.size(), false);
    taken.front() = true;
    taken.back() = true;
    for (size_t j = 1; j + 1 < cap; ++j) {
        const double target = lo + (hi - lo) * static_cast<double>(j) / static_cast<double>(cap - 1);
        size_t best = 0;
        double best_d = std::numeric_limits<double>::infinity();
        for (size_t i = 1; i + 1 < steps.size(); ++i) {
            const double d = std::fabs(x_of(steps[i]) - target);
            if (!taken[i] && d < best_d) {
                best_d = d;
                best = i;
            }
        }
        if (best != 0) taken[best] = true;
    }
    std::vector<RatePoint> ladder;
    for (size_t i = 0; i < steps.size(); ++i)
        if (taken[i]) ladder.push_back(steps[i]);
    return ladder;
}

double cost_per_video_minute(double cpu_seconds, double video_seconds, double price_per_cpu_hour) {
    if (video_seconds <= 0) return 0;
    // r CPU-seconds per video-second => 60 r CPU-seconds per video minute => r/60 CPU-hours.
    const double cpu_hours_per_video_minute = (cpu_seconds / video_seconds) / 60.0;
    return cpu_hours_per_video_minute * price_per_cpu_hour;
}

double break_even_views(double encode_cost_usd, double saved_gb_per_view, double usd_per_gb) {
    const double saving_per_view = saved_gb_per_view * usd_per_gb;
    if (saving_per_view <= 0) return std::numeric_limits<double>::infinity();
    return encode_cost_usd / saving_per_view;
}

}  // namespace ff
