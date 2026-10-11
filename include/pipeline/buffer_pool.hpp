#pragma once
// Fixed-size object pool with blocking acquire.
//
// Frames are preallocated and recycled instead of malloc'd per frame (no
// allocator jitter in the hot path, no copies: ownership just moves between
// stages). Because the pool is finite, an exhausted pool blocks the producer,
// which is the framework's memory-bounded backpressure.

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace ff {

template <class T>
class ObjectPool : public std::enable_shared_from_this<ObjectPool<T>> {
public:
    using Factory = std::function<std::unique_ptr<T>()>;

    static std::shared_ptr<ObjectPool> create(size_t count, const Factory& factory) {
        return std::shared_ptr<ObjectPool>(new ObjectPool(count, factory));
    }

    // Blocks until an object is free. Returns nullptr only after close().
    // The returned shared_ptr hands the object back to the pool when the last
    // reference drops, so stages can fan out references without bookkeeping.
    std::shared_ptr<T> acquire() {
        std::unique_lock lock(mu_);
        if (free_.empty() && !closed_) {
            ++stalls_;
            cv_.wait(lock, [&] { return !free_.empty() || closed_; });
        }
        if (free_.empty()) return nullptr;
        return take_locked();
    }

    std::shared_ptr<T> try_acquire() {
        std::lock_guard lock(mu_);
        if (free_.empty()) return nullptr;
        return take_locked();
    }

    // Wake all blocked acquirers; further acquires on an empty pool return nullptr.
    void close() {
        {
            std::lock_guard lock(mu_);
            closed_ = true;
        }
        cv_.notify_all();
    }

    size_t capacity() const noexcept { return capacity_; }
    size_t available() const {
        std::lock_guard lock(mu_);
        return free_.size();
    }
    // Number of acquire() calls that had to wait for a free object.
    uint64_t stalls() const {
        std::lock_guard lock(mu_);
        return stalls_;
    }

private:
    ObjectPool(size_t count, const Factory& factory) : capacity_(count) {
        free_.reserve(count);
        for (size_t i = 0; i < count; ++i) free_.push_back(factory());
    }

    std::shared_ptr<T> take_locked() {
        std::unique_ptr<T> obj = std::move(free_.back());  // LIFO: most recently used = warmest in cache
        free_.pop_back();
        T* raw = obj.release();
        auto self = this->shared_from_this();
        return std::shared_ptr<T>(raw, [self](T* p) { self->release(p); });
    }

    void release(T* p) {
        {
            std::lock_guard lock(mu_);
            free_.emplace_back(p);
        }
        cv_.notify_one();
    }

    const size_t capacity_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::vector<std::unique_ptr<T>> free_;
    uint64_t stalls_ = 0;
    bool closed_ = false;
};

}  // namespace ff
