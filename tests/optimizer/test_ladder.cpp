#include <algorithm>
#include <cmath>
#include <random>

#include "optimizer/ladder.hpp"
#include "test_util.hpp"

using namespace ff;

namespace {
RatePoint P(int h, int crf, double kbps, double psnr) {
    RatePoint p;
    p.height = h;
    p.width = h * 16 / 9;
    p.crf = crf;
    p.kbps = kbps;
    p.psnr = psnr;
    return p;
}
double x(const RatePoint& p) { return std::log2(p.kbps); }
}  // namespace

TEST(hull_drops_dominated_points) {
    // 1080p at crf 28 costs more than 720p at crf 18 for LESS quality: dominated.
    std::vector<RatePoint> pts = {P(720, 18, 3000, 40.0), P(1080, 28, 3500, 38.0), P(720, 23, 1500, 37.0), P(1080, 18, 9000, 44.0)};
    const auto hull = convex_hull(pts);
    for (const auto& p : hull) CHECK(!(p.height == 1080 && p.crf == 28));
    CHECK_EQ(hull.front().kbps, 1500.0);
}

TEST(hull_is_sorted_increasing_and_concave) {
    std::vector<RatePoint> pts = {P(240, 34, 150, 30), P(240, 23, 400, 34), P(360, 34, 300, 33), P(360, 23, 900, 37),
                                  P(720, 34, 800, 35), P(720, 23, 2500, 41), P(1080, 34, 1800, 36), P(1080, 23, 6000, 44),
                                  P(1080, 18, 11000, 46)};
    const auto h = convex_hull(pts);
    REQUIRE(h.size() >= 3);
    for (size_t i = 1; i < h.size(); ++i) {
        CHECK(h[i].kbps > h[i - 1].kbps);
        CHECK(h[i].psnr > h[i - 1].psnr);
    }
    for (size_t i = 2; i < h.size(); ++i) {
        const double s1 = (h[i - 1].psnr - h[i - 2].psnr) / (x(h[i - 1]) - x(h[i - 2]));
        const double s2 = (h[i].psnr - h[i - 1].psnr) / (x(h[i]) - x(h[i - 1]));
        CHECK(s2 < s1);  // strictly diminishing returns
    }
}

TEST(hull_property_every_point_lies_on_or_below_it) {
    std::mt19937 rng(3);
    for (int trial = 0; trial < 200; ++trial) {
        std::vector<RatePoint> pts;
        const int n = 3 + int(rng() % 25);
        for (int i = 0; i < n; ++i) pts.push_back(P(240, i, 100 + double(rng() % 20000), 25 + double(rng() % 2000) / 100.0));
        const auto h = convex_hull(pts);
        REQUIRE(!h.empty());
        for (const auto& p : pts) {
            if (p.kbps < h.front().kbps || p.kbps > h.back().kbps) continue;
            // interpolate the hull at p.kbps (in log2 domain)
            size_t j = 1;
            while (j + 1 < h.size() && x(h[j]) < x(p)) ++j;
            double q = h.front().psnr;
            if (h.size() >= 2) {
                const double t = (x(p) - x(h[j - 1])) / (x(h[j]) - x(h[j - 1]));
                q = h[j - 1].psnr + t * (h[j].psnr - h[j - 1].psnr);
            }
            CHECK(p.psnr <= q + 1e-9);
        }
        // hull points are a subset of the inputs
        for (const auto& hp : h) CHECK(std::any_of(pts.begin(), pts.end(), [&](const RatePoint& p) { return p.kbps == hp.kbps && p.psnr == hp.psnr; }));
    }
}

TEST(hull_of_single_and_empty_input) {
    CHECK(convex_hull({}).empty());
    CHECK_EQ(convex_hull({P(240, 30, 100, 30)}).size(), size_t(1));
}

TEST(ladder_respects_gap_ratio_and_rung_cap) {
    std::vector<RatePoint> hull;
    for (int i = 0; i < 12; ++i) hull.push_back(P(240 + i, 30 - i, 200.0 * std::pow(1.4, i), 30.0 + 2.0 * i - 0.1 * i * i));
    LadderConfig cfg;
    cfg.max_rungs = 4;
    cfg.min_gap_db = 1.0;
    cfg.min_bitrate_ratio = 1.5;
    cfg.min_db_per_doubling = 0.0;
    const auto l = select_ladder(hull, cfg);
    CHECK(l.size() <= 4);
    REQUIRE(l.size() >= 2);
    for (size_t i = 1; i < l.size(); ++i) {
        CHECK(l[i].psnr - l[i - 1].psnr >= cfg.min_gap_db);
        CHECK(l[i].kbps / l[i - 1].kbps >= cfg.min_bitrate_ratio);
    }
}

TEST(ladder_stops_at_the_knee_where_returns_diminish) {
    // Quality saturates: past 4000 kbps, doubling the rate buys < 1.5 dB.
    std::vector<RatePoint> pts = {P(360, 30, 500, 34), P(720, 26, 1000, 38), P(1080, 22, 2000, 41.5), P(1080, 20, 4000, 44), P(1080, 18, 8000, 45.2),
                                  P(1080, 16, 16000, 45.8)};
    LadderConfig cfg;
    cfg.min_gap_db = 0.5;
    cfg.min_bitrate_ratio = 1.5;
    cfg.min_db_per_doubling = 1.5;
    const auto l = select_ladder(pts, cfg);
    REQUIRE(!l.empty());
    for (const auto& r : l) CHECK(r.kbps <= 4000.0);
    CHECK(l.back().kbps == 4000.0);  // the ladder reaches the knee
}

TEST(ladder_uses_frontier_points_that_sit_below_the_hull) {
    // The 1500 kbps probe lies under the 500 -> 6000 hull segment, so it is not on
    // the hull, but it is a real, non-dominated operating point and fills the gap.
    std::vector<RatePoint> pts = {P(240, 30, 500, 34.0), P(480, 26, 1500, 36.5), P(1080, 20, 6000, 44.0)};
    CHECK_EQ(convex_hull(pts).size(), size_t(2));
    LadderConfig cfg;
    cfg.min_db_per_doubling = 0.5;
    const auto l = select_ladder(pts, cfg);
    CHECK_EQ(l.size(), size_t(3));
}

TEST(ladder_applies_bitrate_bounds) {
    std::vector<RatePoint> pts = {P(240, 34, 100, 30), P(360, 30, 400, 35), P(720, 24, 1600, 40), P(1080, 20, 6400, 44)};
    LadderConfig cfg;
    cfg.min_kbps = 300;
    cfg.max_kbps = 2000;
    cfg.min_db_per_doubling = 0;
    const auto l = select_ladder(pts, cfg);
    CHECK(!l.empty());
    for (const auto& r : l) {
        CHECK(r.kbps >= 300);
        CHECK(r.kbps <= 2000);
    }
}

TEST(ladder_over_cap_is_thinned_but_keeps_both_ends) {
    std::vector<RatePoint> pts;
    for (int i = 0; i < 20; ++i) pts.push_back(P(240 + i, 40 - i, 100.0 * std::pow(1.5, i), 30.0 + 1.2 * i));
    LadderConfig cfg;
    cfg.max_rungs = 5;
    cfg.min_gap_db = 1.0;
    cfg.min_bitrate_ratio = 1.2;
    cfg.min_db_per_doubling = 0.0;
    const auto l = select_ladder(pts, cfg);
    REQUIRE(l.size() == 5);
    CHECK_NEAR(l.front().kbps, 100.0, 1e-9);                      // cheapest point kept
    CHECK_NEAR(l.back().kbps, 100.0 * std::pow(1.5, 19), 1e-6);   // top of the curve reached, not cut off at rung 5
    for (size_t i = 1; i < l.size(); ++i) {
        CHECK(l[i].psnr - l[i - 1].psnr >= cfg.min_gap_db);
        CHECK(l[i].kbps / l[i - 1].kbps >= cfg.min_bitrate_ratio);
    }
}

TEST(ladder_of_nothing_is_empty) {
    CHECK(select_ladder({}, LadderConfig{}).empty());
}

TEST(ladder_is_strictly_increasing_for_random_probe_sets) {
    std::mt19937 rng(21);
    for (int trial = 0; trial < 200; ++trial) {
        std::vector<RatePoint> pts;
        for (int i = 0; i < 30; ++i) pts.push_back(P(240, i, 50 + double(rng() % 9000), 28 + double(rng() % 2000) / 100.0));
        const auto l = select_ladder(pts, LadderConfig{});
        for (size_t i = 1; i < l.size(); ++i) {
            CHECK(l[i].kbps > l[i - 1].kbps);
            CHECK(l[i].psnr > l[i - 1].psnr);
        }
        CHECK(l.size() <= 6);
    }
}

TEST(cost_model) {
    // 60 CPU-seconds spent on a 30 s video = 2 CPU-s per video-s = 120 CPU-s per video minute = 1/30 CPU-hour.
    CHECK_NEAR(cost_per_video_minute(60, 30, 3.6), 3.6 / 30.0, 1e-12);
    CHECK_NEAR(break_even_views(0.12, 0.001, 0.02), 6000.0, 1e-6);  // saves $0.00002 per view
    CHECK(std::isinf(break_even_views(1.0, 0.0, 0.02)));
}
