// Ring buffer vs a mutex queue, buffer pool, and end-to-end pipeline throughput.

#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <thread>

#include "bench_util.hpp"
#include "kernels/dispatch.hpp"
#include "kernels/luma.hpp"
#include "kernels/resize.hpp"
#include "pipeline/pipeline.hpp"
#include "pipeline/ring_buffer.hpp"

using namespace ff;

namespace {

// Baseline: the obvious bounded queue guarded by a mutex + condition variables.
template <class T>
class MutexQueue {
public:
    explicit MutexQueue(size_t cap) : cap_(cap) {}
    void push(T v) {
        std::unique_lock lk(m_);
        not_full_.wait(lk, [&] { return q_.size() < cap_; });
        q_.push_back(std::move(v));
        not_empty_.notify_one();
    }
    T pop() {
        std::unique_lock lk(m_);
        not_empty_.wait(lk, [&] { return !q_.empty(); });
        T v = std::move(q_.front());
        q_.pop_front();
        not_full_.notify_one();
        return v;
    }

private:
    size_t cap_;
    std::mutex m_;
    std::condition_variable not_full_, not_empty_;
    std::deque<T> q_;
};

double mops_ring(size_t items, size_t cap) {
    Channel<uint64_t> ch(cap);
    const auto t0 = std::chrono::steady_clock::now();
    std::thread prod([&] {
        for (uint64_t i = 0; i < items; ++i) ch.push(uint64_t(i));
        ch.close();
    });
    uint64_t sum = 0;
    while (auto v = ch.pop()) sum += *v;
    prod.join();
    bench::g_sink = bench::g_sink + sum;
    return static_cast<double>(items) / std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 1e6;
}

double mops_mutex(size_t items, size_t cap) {
    MutexQueue<uint64_t> q(cap);
    const auto t0 = std::chrono::steady_clock::now();
    std::thread prod([&] {
        for (uint64_t i = 0; i < items; ++i) q.push(uint64_t(i));
    });
    uint64_t sum = 0;
    for (size_t i = 0; i < items; ++i) sum += q.pop();
    prod.join();
    bench::g_sink = bench::g_sink + sum;
    return static_cast<double>(items) / std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 1e6;
}

}  // namespace

int main() {
    std::printf("# Pipeline benchmarks (SIMD: %s, %u hardware threads)\n", to_string(kernels().isa),
                std::thread::hardware_concurrency());

    bench::header("SPSC queue, 2 threads, 8-byte items");
    std::printf("| capacity | lock-free ring (Mitems/s) | mutex + condvar (Mitems/s) | ratio |\n|---:|---:|---:|---:|\n");
    for (size_t cap : {4, 64, 1024}) {
        double r = 0, m = 0;
        for (int rep = 0; rep < 3; ++rep) {  // best of 3
            r = std::max(r, mops_ring(1'000'000, cap));
            m = std::max(m, mops_mutex(1'000'000, cap));
        }
        std::printf("| %zu | %.1f | %.1f | %.1fx |\n", cap, r, m, r / m);
    }

    bench::header("Frame pool");
    {
        auto pool = make_frame_pool(1920, 1080, 4);
        const double t_pool = bench::seconds_per_call([&] {
            FramePtr f = pool->acquire();
            return uint64_t(f->pts);
        });
        const double t_alloc = bench::seconds_per_call([&] {
            auto f = std::make_shared<Frame>(1920, 1080);  // allocate + zero 3 MB every time
            return uint64_t(f->pts);
        });
        std::printf("| acquire from pool | allocate a fresh 1080p frame | ratio |\n|---:|---:|---:|\n");
        std::printf("| %.0f ns | %.0f us | %.0fx |\n", t_pool * 1e9, t_alloc * 1e6, t_alloc / t_pool);
    }

    bench::header("Frame pipeline: source -> luma-mean -> 2x downscale -> sink (1080p frames, 600 frames, pool of 6)");
    {
        const int N = 600;
        const Isa original = kernels().isa;
        std::vector<uint8_t> small(960 * 540);
        auto run = [&](size_t depth, bool verbose) {
            auto pool = make_frame_pool(1920, 1080, 6);
            FramePipeline p(depth);
            double acc = 0;
            p.add_stage("luma-mean", [&](FramePtr f) {
                acc += luma_mean(f->data(0), f->stride(0), f->width(), f->height());
                return f;
            });
            p.add_stage("downscale2x", [&](FramePtr f) {
                downscale2x(f->data(0), f->stride(0), f->width(), f->height(), small.data(), 960);
                acc += small[7];
                return f;
            });
            int next = 0;
            const auto stats = p.run(
                [&]() -> FramePtr {
                    if (next >= N) return nullptr;
                    FramePtr f = pool->acquire();
                    f->pts = next++;
                    return f;
                },
                [](FramePtr) {});
            bench::g_sink = bench::g_sink + static_cast<uint64_t>(acc);
            if (verbose) {
                std::printf("\nper-stage counters (queue depth %zu, %s):\n", depth, to_string(kernels().isa));
                for (const auto& s : stats.stages)
                    std::printf("  %-12s busy %5.1f%%  in-stalls %-6llu out-stalls %-6llu\n", s.name.c_str(),
                                100.0 * s.busy_seconds / stats.wall_seconds, static_cast<unsigned long long>(s.in_stalls),
                                static_cast<unsigned long long>(s.out_stalls));
            }
            return N / stats.wall_seconds;
        };
        std::printf("| SIMD backend | queue depth 1 (fps) | queue depth 4 (fps) |\n|---|---:|---:|\n");
        for (Isa isa : supported_isas()) {
            set_active_isa(isa);
            double d1 = 0, d4 = 0;
            for (int rep = 0; rep < 3; ++rep) {
                d1 = std::max(d1, run(1, false));
                d4 = std::max(d4, run(4, false));
            }
            std::printf("| %s | %.0f | %.0f |\n", to_string(isa), d1, d4);
        }
        set_active_isa(original);
        run(4, true);
    }
    return 0;
}
