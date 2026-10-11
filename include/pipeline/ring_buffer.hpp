#pragma once
// Single-producer / single-consumer lock-free ring buffer, and a blocking
// Channel built on it.
//
// Design notes (all of these are concepts from the transcoding notes):
//  * Only two atomics: `head_` (written by the consumer) and `tail_` (written
//    by the producer). No CAS loop, so no ABA problem and the operations are
//    wait-free for their single owner.
//  * Memory ordering: the producer writes the slot, then publishes it with a
//    release store to `tail_`. The consumer's acquire load of `tail_` that
//    observes the new value therefore also observes the slot contents. The same
//    pairing, mirrored, lets the producer safely reuse a slot.
//  * False sharing: the producer-owned and consumer-owned state live on
//    separate 64-byte cache lines.
//  * Cached opposite index: each side re-reads the other side's index only when
//    its cached copy says "full"/"empty", which keeps the shared lines quiet
//    on the fast path.

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <thread>
#include <utility>

namespace ff {

inline constexpr size_t kCacheLine = 64;

inline void cpu_relax() noexcept {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield");
#endif
}

// Spin, then yield, then sleep: low latency when the wait is short, no burned
// core when a stage is genuinely starved.
class Backoff {
public:
    void wait() noexcept {
        if (n_ < 64) {
            cpu_relax();
        } else if (n_ < 192) {
            std::this_thread::yield();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        if (n_ < 1000) ++n_;
    }
    void reset() noexcept { n_ = 0; }

private:
    int n_ = 0;
};

// Exactly one thread may call try_push and exactly one (other) thread try_pop.
template <class T>
class SpscRingBuffer {
public:
    explicit SpscRingBuffer(size_t min_capacity)
        : capacity_(std::bit_ceil(std::max<size_t>(min_capacity, 2))),
          mask_(capacity_ - 1),
          slots_(new T[capacity_]) {}

    SpscRingBuffer(const SpscRingBuffer&) = delete;
    SpscRingBuffer& operator=(const SpscRingBuffer&) = delete;

    size_t capacity() const noexcept { return capacity_; }

    // Moves from `v` only on success, so the caller still owns it on failure.
    bool try_push(T&& v) {
        const size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail - head_cache_ == capacity_) {
            head_cache_ = head_.load(std::memory_order_acquire);
            if (tail - head_cache_ == capacity_) return false;
        }
        slots_[tail & mask_] = std::move(v);
        tail_.store(tail + 1, std::memory_order_release);
        return true;
    }

    bool try_pop(T& out) {
        const size_t head = head_.load(std::memory_order_relaxed);
        if (head == tail_cache_) {
            tail_cache_ = tail_.load(std::memory_order_acquire);
            if (head == tail_cache_) return false;
        }
        out = std::move(slots_[head & mask_]);
        head_.store(head + 1, std::memory_order_release);
        return true;
    }

    // Racy snapshot, for stats only.
    size_t size_approx() const noexcept {
        return tail_.load(std::memory_order_relaxed) - head_.load(std::memory_order_relaxed);
    }

private:
    // Read-only after construction (shares the first line, never written).
    const size_t capacity_;
    const size_t mask_;
    std::unique_ptr<T[]> slots_;

    // Consumer-owned line.
    alignas(kCacheLine) std::atomic<size_t> head_{0};
    size_t tail_cache_ = 0;

    // Producer-owned line.
    alignas(kCacheLine) std::atomic<size_t> tail_{0};
    size_t head_cache_ = 0;
};

// Bounded, blocking, single-producer/single-consumer channel with end-of-stream
// and cancellation. A full channel blocks the producer: that is backpressure.
template <class T>
class Channel {
public:
    explicit Channel(size_t capacity) : ring_(capacity) {}

    // Producer side. Returns false if the channel was cancelled.
    bool push(T v) {
        Backoff backoff;
        bool stalled = false;
        if (cancelled_.load(std::memory_order_acquire)) return false;
        while (!ring_.try_push(std::move(v))) {
            if (cancelled_.load(std::memory_order_acquire)) return false;
            if (!stalled) {
                push_stalls_.fetch_add(1, std::memory_order_relaxed);
                stalled = true;
            }
            backoff.wait();
        }
        return true;
    }

    // Consumer side. nullopt means end-of-stream (closed and drained) or cancelled.
    std::optional<T> pop() {
        Backoff backoff;
        bool stalled = false;
        T out{};
        for (;;) {
            if (ring_.try_pop(out)) return std::optional<T>(std::move(out));
            if (cancelled_.load(std::memory_order_acquire)) return std::nullopt;
            if (closed_.load(std::memory_order_acquire)) {
                // The producer pushed everything before closing: drain once more.
                if (ring_.try_pop(out)) return std::optional<T>(std::move(out));
                return std::nullopt;
            }
            if (!stalled) {
                pop_stalls_.fetch_add(1, std::memory_order_relaxed);
                stalled = true;
            }
            backoff.wait();
        }
    }

    // Consumer side: drop whatever is still queued (releases pooled frames promptly after a cancel).
    void drain() {
        T tmp{};
        while (ring_.try_pop(tmp)) tmp = T{};
    }

    void close() noexcept { closed_.store(true, std::memory_order_release); }
    void cancel() noexcept { cancelled_.store(true, std::memory_order_release); }

    uint64_t push_stalls() const noexcept { return push_stalls_.load(std::memory_order_relaxed); }
    uint64_t pop_stalls() const noexcept { return pop_stalls_.load(std::memory_order_relaxed); }
    size_t capacity() const noexcept { return ring_.capacity(); }

private:
    SpscRingBuffer<T> ring_;
    std::atomic<bool> closed_{false};
    std::atomic<bool> cancelled_{false};
    std::atomic<uint64_t> push_stalls_{0};
    std::atomic<uint64_t> pop_stalls_{0};
};

}  // namespace ff
