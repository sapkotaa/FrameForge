#include "pipeline/pipeline.hpp"

#include <chrono>
#include <memory>
#include <mutex>
#include <thread>

namespace ff {
namespace {
using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }
}  // namespace

PipelineStats FramePipeline::run(Source source, Sink sink, std::function<void()> on_cancel) {
    const size_t n = stages_.size();
    std::vector<std::unique_ptr<Channel<FramePtr>>> ch;
    for (size_t i = 0; i <= n; ++i) ch.push_back(std::make_unique<Channel<FramePtr>>(depth_));

    PipelineStats result;
    result.stages.resize(n + 2);  // each thread writes only its own element
    result.stages.front().name = "source";
    for (size_t i = 0; i < n; ++i) result.stages[i + 1].name = stages_[i].name;
    result.stages.back().name = "sink";

    std::exception_ptr first_error;
    std::mutex err_mu;
    auto fail = [&](std::exception_ptr e) {
        bool first = false;
        {
            std::lock_guard lock(err_mu);
            if (!first_error) {
                first_error = e;
                first = true;
            }
        }
        for (auto& c : ch) c->cancel();
        if (first && on_cancel) on_cancel();
    };

    const auto t_start = Clock::now();
    std::vector<std::thread> threads;

    threads.emplace_back([&] {
        StageStats& s = result.stages[0];
        try {
            for (;;) {
                const auto t0 = Clock::now();
                FramePtr f = source();
                s.busy_seconds += seconds_since(t0);
                if (!f) break;
                ++s.frames_out;
                if (!ch[0]->push(std::move(f))) break;
            }
            ch[0]->close();
        } catch (...) {
            fail(std::current_exception());
        }
        s.out_stalls = ch[0]->push_stalls();
    });

    for (size_t i = 0; i < n; ++i) {
        threads.emplace_back([&, i] {
            StageStats& s = result.stages[i + 1];
            try {
                while (auto item = ch[i]->pop()) {
                    ++s.frames_in;
                    const auto t0 = Clock::now();
                    FramePtr out = stages_[i].fn(std::move(*item));
                    s.busy_seconds += seconds_since(t0);
                    if (out) {
                        ++s.frames_out;
                        if (!ch[i + 1]->push(std::move(out))) break;
                    }
                }
                ch[i + 1]->close();
            } catch (...) {
                fail(std::current_exception());
            }
            ch[i]->drain();
            s.in_stalls = ch[i]->pop_stalls();
            s.out_stalls = ch[i + 1]->push_stalls();
        });
    }

    StageStats& sk = result.stages.back();
    try {
        while (auto item = ch[n]->pop()) {
            ++sk.frames_in;
            const auto t0 = Clock::now();
            sink(std::move(*item));
            sk.busy_seconds += seconds_since(t0);
        }
    } catch (...) {
        fail(std::current_exception());
    }
    ch[n]->drain();
    sk.in_stalls = ch[n]->pop_stalls();

    for (auto& t : threads) t.join();
    result.wall_seconds = seconds_since(t_start);
    if (first_error) std::rethrow_exception(first_error);
    return result;
}

}  // namespace ff
