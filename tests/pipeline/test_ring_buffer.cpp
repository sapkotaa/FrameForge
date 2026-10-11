#include <atomic>
#include <thread>

#include "pipeline/ring_buffer.hpp"
#include "test_util.hpp"

using namespace ff;

TEST(ring_capacity_rounds_up_to_power_of_two) {
    SpscRingBuffer<int> r(5);
    CHECK_EQ(r.capacity(), size_t(8));
    SpscRingBuffer<int> r2(1);
    CHECK_EQ(r2.capacity(), size_t(2));
}

TEST(ring_fifo_order_full_and_empty) {
    SpscRingBuffer<int> r(4);
    int v = 0;
    CHECK(!r.try_pop(v));
    for (int i = 0; i < 4; ++i) CHECK(r.try_push(int(i)));
    CHECK(!r.try_push(99));  // full
    for (int i = 0; i < 4; ++i) {
        CHECK(r.try_pop(v));
        CHECK_EQ(v, i);
    }
    CHECK(!r.try_pop(v));
}

TEST(ring_wraps_around_many_times) {
    SpscRingBuffer<uint64_t> r(8);
    uint64_t next_in = 0, next_out = 0, v = 0;
    for (int round = 0; round < 1000; ++round) {
        for (int i = 0; i < 5; ++i) CHECK(r.try_push(uint64_t(next_in++)));
        for (int i = 0; i < 5; ++i) {
            CHECK(r.try_pop(v));
            CHECK_EQ(v, next_out++);
        }
    }
}

TEST(ring_failed_push_does_not_consume_the_value) {
    SpscRingBuffer<std::shared_ptr<int>> r(2);
    CHECK(r.try_push(std::make_shared<int>(1)));
    CHECK(r.try_push(std::make_shared<int>(2)));
    auto keep = std::make_shared<int>(3);
    CHECK(!r.try_push(std::move(keep)));
    CHECK(keep != nullptr);  // still ours: try_push moves only on success
}

// Two real threads hammering the ring. Run under -fsanitize=thread to check
// the acquire/release pairing; here we check that nothing is lost, duplicated
// or reordered, and that the payload written before the publish is visible.
TEST(ring_spsc_stress_preserves_order_and_payload) {
    constexpr uint64_t kItems = 3'000'000;
    struct Item {
        uint64_t seq;
        uint64_t check;  // written before publication, must be visible after consumption
    };
    SpscRingBuffer<Item> r(1024);
    std::atomic<bool> bad{false};

    std::thread producer([&] {
        for (uint64_t i = 0; i < kItems; ++i) {
            Item it{i, i * 0x9E3779B97F4A7C15ull};
            while (!r.try_push(std::move(it))) cpu_relax();
        }
    });
    std::thread consumer([&] {
        Item it{};
        for (uint64_t expect = 0; expect < kItems; ++expect) {
            while (!r.try_pop(it)) cpu_relax();
            if (it.seq != expect || it.check != expect * 0x9E3779B97F4A7C15ull) {
                bad = true;
                return;
            }
        }
    });
    producer.join();
    consumer.join();
    CHECK(!bad.load());
}

TEST(channel_delivers_everything_then_signals_end) {
    Channel<int> ch(4);
    std::thread prod([&] {
        for (int i = 0; i < 10000; ++i) ch.push(int(i));
        ch.close();
    });
    int expect = 0;
    while (auto v = ch.pop()) CHECK_EQ(*v, expect++);
    prod.join();
    CHECK_EQ(expect, 10000);
}

TEST(channel_full_queue_blocks_the_producer) {
    Channel<int> ch(2);
    std::atomic<int> pushed{0};
    std::thread prod([&] {
        for (int i = 0; i < 100; ++i) {
            if (!ch.push(int(i))) break;
            ++pushed;
        }
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK(pushed.load() <= 2);  // capacity 2: backpressure held it back
    CHECK(ch.push_stalls() >= 1);
    ch.cancel();  // release the producer
    prod.join();
}

TEST(channel_cancel_wakes_a_blocked_consumer) {
    Channel<int> ch(2);
    std::thread cons([&] { CHECK(!ch.pop().has_value()); });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    ch.cancel();
    cons.join();
}
