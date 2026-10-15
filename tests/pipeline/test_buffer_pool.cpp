#include <atomic>
#include <thread>

#include "pipeline/buffer_pool.hpp"
#include "pipeline/frame.hpp"
#include "test_util.hpp"

using namespace ff;

TEST(frames_are_aligned_and_zeroed) {
    Frame f(100, 50);
    for (int p = 0; p < 3; ++p) {
        CHECK(reinterpret_cast<uintptr_t>(f.data(p)) % 64 == 0);
        CHECK(f.stride(p) % 64 == 0);
        CHECK_EQ(int(f.data(p)[0]), 0);
    }
    CHECK_EQ(f.byte_size(), size_t(100 * 50 * 3 / 2));
}



TEST(frame_rejects_odd_dimensions) {
    bool threw = false;
    try {
        Frame f(101, 50);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

TEST(pool_recycles_objects) {
    auto pool = make_frame_pool(16, 16, 2);
    CHECK_EQ(pool->available(), size_t(2));
    Frame* addr = nullptr;
    {
        FramePtr a = pool->acquire();
        addr = a.get();
        CHECK_EQ(pool->available(), size_t(1));
    }
    CHECK_EQ(pool->available(), size_t(2));
    FramePtr b = pool->acquire();
    CHECK(b.get() == addr);  // LIFO reuse of the warm frame
}

TEST(pool_exhaustion_blocks_until_release) {
    auto pool = make_frame_pool(16, 16, 1);
    FramePtr held = pool->acquire();
    CHECK(pool->try_acquire() == nullptr);

    std::atomic<bool> got{false};
    std::thread t([&] {
        FramePtr f = pool->acquire();  // must wait
        got = (f != nullptr);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK(!got.load());
    held.reset();  // releasing wakes the waiter
    t.join();
    CHECK(got.load());
    CHECK(pool->stalls() >= 1);
}

TEST(pool_close_unblocks_waiters) {
    auto pool = make_frame_pool(16, 16, 1);
    FramePtr held = pool->acquire();
    std::atomic<bool> null_result{false};
    std::thread t([&] { null_result = (pool->acquire() == nullptr); });
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    pool->close();
    t.join();
    CHECK(null_result.load());
}

TEST(frames_outlive_the_pool_handle) {
    FramePtr f;
    {
        auto pool = make_frame_pool(16, 16, 1);
        f = pool->acquire();
    }  // pool handle gone; the frame keeps the pool alive until it is released
    f->fill(1, 2, 3);
    CHECK_EQ(int(f->data(0)[0]), 1);
}

TEST(pool_is_thread_safe_under_contention) {
    auto pool = make_frame_pool(32, 32, 4);
    std::atomic<int> total{0};
    std::vector<std::thread> ts;
    for (int t = 0; t < 4; ++t)
        ts.emplace_back([&] {
            for (int i = 0; i < 2000; ++i) {
                FramePtr f = pool->acquire();
                f->fill(uint8_t(i), 0, 0);
                ++total;
            }
        });
    for (auto& t : ts) t.join();
    CHECK_EQ(total.load(), 8000);
    CHECK_EQ(pool->available(), size_t(4));
}



