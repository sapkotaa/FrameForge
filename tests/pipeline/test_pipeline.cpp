#include <atomic>
#include <thread>

#include "pipeline/pipeline.hpp"
#include "test_util.hpp"

using namespace ff;

namespace {
FramePtr numbered(const std::shared_ptr<FramePool>& pool, int i) {
    FramePtr f = pool->acquire();
    if (f) f->pts = i;  // null once the pool is closed during teardown
    return f;
}
}  // namespace

TEST(pipeline_preserves_order_through_stages) {
    auto pool = make_frame_pool(16, 16, 8);
    FramePipeline p(2);
    p.add_stage("double", [](FramePtr f) { f->pts *= 2; return f; });
    p.add_stage("plus1", [](FramePtr f) { f->pts += 1; return f; });
    int next = 0;
    std::vector<int64_t> got;
    auto stats = p.run([&]() -> FramePtr { return next < 500 ? numbered(pool, next++) : nullptr; },
                       [&](FramePtr f) { got.push_back(f->pts); });
    REQUIRE(got.size() == 500);
    for (int i = 0; i < 500; ++i) CHECK_EQ(got[static_cast<size_t>(i)], int64_t(i) * 2 + 1);
    CHECK_EQ(stats.stages.size(), size_t(4));  // source, 2 stages, sink
    CHECK_EQ(stats.stages[1].frames_in, uint64_t(500));
    CHECK_EQ(stats.stages[3].frames_in, uint64_t(500));
}

TEST(pipeline_stage_can_drop_frames) {
    auto pool = make_frame_pool(16, 16, 8);
    FramePipeline p(2);
    p.add_stage("evens", [](FramePtr f) -> FramePtr { return f->pts % 2 == 0 ? f : nullptr; });
    int next = 0, kept = 0;
    p.run([&]() -> FramePtr { return next < 100 ? numbered(pool, next++) : nullptr; }, [&](FramePtr) { ++kept; });
    CHECK_EQ(kept, 50);
}

TEST(pipeline_slow_sink_throttles_the_source) {
    // Pool of 3 frames, queues of depth 2: no matter how fast the source runs it
    // can never have more than 3 frames in flight, because acquire() blocks.
    auto pool = make_frame_pool(16, 16, 3);
    FramePipeline p(2);
    p.add_stage("pass", [](FramePtr f) { return f; });
    std::atomic<int> in_flight{0}, max_in_flight{0};
    int next = 0;
    auto stats = p.run(
        [&]() -> FramePtr {
            if (next >= 60) return nullptr;
            FramePtr f = numbered(pool, next++);
            const int now = ++in_flight;
            int prev = max_in_flight.load();
            while (now > prev && !max_in_flight.compare_exchange_weak(prev, now)) {}
            return f;
        },
        [&](FramePtr) {
            std::this_thread::sleep_for(std::chrono::microseconds(300));
            --in_flight;
        });
    CHECK(max_in_flight.load() <= 3);
    CHECK(pool->stalls() > 0);          // the source did have to wait for the sink
    CHECK(stats.stages[0].out_stalls > 0 || pool->stalls() > 0);
    CHECK_EQ(pool->available(), size_t(3));  // everything returned
}

TEST(pipeline_exception_in_a_stage_is_rethrown_and_nothing_hangs) {
    auto pool = make_frame_pool(16, 16, 4);
    FramePipeline p(2);
    p.add_stage("boom", [](FramePtr f) -> FramePtr {
        if (f->pts == 7) throw std::runtime_error("stage failed");
        return f;
    });
    int next = 0;
    bool threw = false;
    try {
        p.run([&]() -> FramePtr { return numbered(pool, next++); }, [](FramePtr) {}, [&] { pool->close(); });
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()) == "stage failed";
    }
    CHECK(threw);
}

TEST(pipeline_exception_in_the_sink_cancels_upstream) {
    auto pool = make_frame_pool(16, 16, 4);
    FramePipeline p(2);
    p.add_stage("pass", [](FramePtr f) { return f; });
    int next = 0;
    bool threw = false;
    try {
        p.run([&]() -> FramePtr { return numbered(pool, next++); },
              [](FramePtr f) { if (f->pts == 5) throw std::runtime_error("sink failed"); }, [&] { pool->close(); });
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

TEST(pipeline_reports_busy_time_per_stage) {
    auto pool = make_frame_pool(16, 16, 4);
    FramePipeline p(2);
    p.add_stage("slow", [](FramePtr f) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        return f;
    });
    int next = 0;
    auto stats = p.run([&]() -> FramePtr { return next < 20 ? numbered(pool, next++) : nullptr; }, [](FramePtr) {});
    CHECK(stats.stages[1].busy_seconds >= 0.035);                         // ~20 x 2 ms
    CHECK(stats.stages[1].busy_seconds > stats.stages[0].busy_seconds);   // the bottleneck stands out
}
