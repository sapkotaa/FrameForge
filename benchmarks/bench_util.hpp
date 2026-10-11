#pragma once
// Minimal benchmark harness: warm up, time batches, report the best (least
// noisy) per-call time. Results are folded into a volatile sink so the
// compiler cannot delete the work.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace bench {

inline volatile uint64_t g_sink = 0;

template <class F>
double seconds_per_call(F&& f, double min_batch_seconds = 0.05, int batches = 7) {
    using Clock = std::chrono::steady_clock;
    for (int i = 0; i < 3; ++i) g_sink = g_sink + f();  // warm-up: caches, branch predictors, frequency ramp
    // Pick an iteration count so one batch lasts ~min_batch_seconds.
    size_t iters = 1;
    for (;;) {
        const auto t0 = Clock::now();
        for (size_t i = 0; i < iters; ++i) g_sink = g_sink + f();
        const double dt = std::chrono::duration<double>(Clock::now() - t0).count();
        if (dt >= min_batch_seconds / 4 || iters > (1u << 28)) break;
        iters *= 2;
    }
    iters = std::max<size_t>(1, static_cast<size_t>(static_cast<double>(iters) * 4));
    double best = 1e30;
    for (int b = 0; b < batches; ++b) {
        const auto t0 = Clock::now();
        for (size_t i = 0; i < iters; ++i) g_sink = g_sink + f();
        const double dt = std::chrono::duration<double>(Clock::now() - t0).count() / static_cast<double>(iters);
        best = std::min(best, dt);
    }
    return best;
}

inline void header(const char* title) { std::printf("\n## %s\n\n", title); }

}  // namespace bench
