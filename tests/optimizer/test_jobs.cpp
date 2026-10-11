#include <atomic>
#include <chrono>
#include <set>
#include <stdexcept>
#include <thread>

#include "optimizer/chunked.hpp"
#include "optimizer/job_runner.hpp"
#include "test_util.hpp"

using namespace ff;

TEST(runner_executes_every_job) {
    JobRunner r(4);
    std::atomic<int> n{0};
    std::vector<std::future<void>> fs;
    for (int i = 0; i < 200; ++i) fs.push_back(r.submit("job", [&] { ++n; }));
    for (auto& f : fs) f.get();
    r.wait_idle();
    CHECK_EQ(n.load(), 200);
    const auto s = r.stats();
    CHECK_EQ(s.submitted, uint64_t(200));
    CHECK_EQ(s.completed, uint64_t(200));
    CHECK_EQ(s.failed, uint64_t(0));
}

TEST(runner_retries_a_flaky_job_until_it_succeeds) {
    JobRunner r(2);
    std::atomic<int> attempts{0};
    auto f = r.submit("flaky", [&] {
        if (++attempts < 3) throw std::runtime_error("preempted");
    }, /*max_attempts=*/3);
    f.get();  // must not throw
    CHECK_EQ(attempts.load(), 3);
    r.wait_idle();
    CHECK_EQ(r.stats().retries, uint64_t(2));
    CHECK_EQ(r.stats().failed, uint64_t(0));
}

TEST(runner_surfaces_a_job_that_always_fails) {
    JobRunner r(2);
    auto f = r.submit("doomed", [] { throw std::runtime_error("disk full"); }, 2);
    bool threw = false;
    try {
        f.get();
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()) == "disk full";
    }
    CHECK(threw);
    r.wait_idle();
    CHECK_EQ(r.stats().failed, uint64_t(1));
    CHECK_EQ(r.stats().retries, uint64_t(1));
}

TEST(retried_idempotent_jobs_leave_consistent_results) {
    // Each job overwrites its own slot; a failure after a partial write followed by a retry must still end clean.
    JobRunner r(4);
    std::vector<std::vector<int>> slots(16);
    std::vector<std::atomic<int>> tries(16);
    std::vector<std::future<void>> fs;
    for (int i = 0; i < 16; ++i)
        fs.push_back(r.submit("idem", [&, i] {
            slots[static_cast<size_t>(i)].clear();  // rebuild from scratch on every attempt
            for (int k = 0; k < 10; ++k) slots[static_cast<size_t>(i)].push_back(i * 100 + k);
            if (tries[static_cast<size_t>(i)]++ == 0 && i % 2 == 0) throw std::runtime_error("worker lost");  // fail AFTER writing
        }));
    for (auto& f : fs) f.get();
    for (int i = 0; i < 16; ++i) {
        REQUIRE(slots[static_cast<size_t>(i)].size() == 10);
        CHECK_EQ(slots[static_cast<size_t>(i)].back(), i * 100 + 9);
    }
}

TEST(runner_runs_jobs_in_parallel) {
    JobRunner r(4);
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::future<void>> fs;
    for (int i = 0; i < 8; ++i) fs.push_back(r.submit("sleep", [] { std::this_thread::sleep_for(std::chrono::milliseconds(60)); }));
    for (auto& f : fs) f.get();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    CHECK(s < 0.35);  // serial would take 0.48 s; 4 workers need ~0.12 s
    CHECK(r.stats().busy_seconds > 0.4);
}

TEST(runner_destructor_finishes_queued_work) {
    std::atomic<int> n{0};
    {
        JobRunner r(1);
        for (int i = 0; i < 20; ++i) r.submit("q", [&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            ++n;
        });
    }
    CHECK_EQ(n.load(), 20);
}

// ------------------------------------------------------------------ chunk planning

namespace {
void check_plan(const std::vector<Chunk>& plan, size_t n, size_t min_len, size_t max_len) {
    size_t pos = 0;
    for (size_t i = 0; i < plan.size(); ++i) {
        CHECK_EQ(plan[i].start, pos);  // contiguous, no gaps or overlaps
        CHECK(plan[i].count >= 1);
        CHECK(plan[i].count <= max_len);
        if (i + 1 < plan.size()) CHECK(plan[i].count >= min_len);
        pos += plan[i].count;
    }
    CHECK_EQ(pos, n);
}
}  // namespace

TEST(plan_without_cuts_uses_max_length) {
    const auto p = plan_chunks(100, {}, 10, 30);
    check_plan(p, 100, 10, 30);
    CHECK_EQ(p.size(), size_t(4));  // 30,30,30,10
}

TEST(plan_splits_at_scene_cuts) {
    const auto p = plan_chunks(100, {25, 60}, 10, 50);
    check_plan(p, 100, 10, 50);
    REQUIRE(p.size() == 3);
    CHECK_EQ(p[0].count, size_t(25));
    CHECK_EQ(p[1].start, size_t(25));
    CHECK_EQ(p[2].start, size_t(60));
}

TEST(plan_ignores_cuts_that_would_make_chunks_too_short) {
    const auto p = plan_chunks(100, {3, 12, 15}, 10, 50);  // cut at 3 and 15 are < 10 frames after the previous boundary
    check_plan(p, 100, 10, 50);
    CHECK_EQ(p[0].count, size_t(12));
}

TEST(plan_invariants_hold_for_random_inputs) {
    std::mt19937 rng(11);
    for (int t = 0; t < 500; ++t) {
        const size_t n = 1 + rng() % 400;
        const size_t mn = 1 + rng() % 20, mx = mn + rng() % 60;
        std::vector<int> cuts;
        for (int c = 1; c < static_cast<int>(n); ++c)
            if (rng() % 17 == 0) cuts.push_back(c);
        check_plan(plan_chunks(n, cuts, mn, mx), n, mn, mx);
    }
}

TEST(plan_rejects_bad_bounds) {
    bool threw = false;
    try {
        plan_chunks(10, {}, 0, 5);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}
