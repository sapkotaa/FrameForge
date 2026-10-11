#include "optimizer/job_runner.hpp"

#include <algorithm>
#include <chrono>

namespace ff {

JobRunner::JobRunner(unsigned workers) {
    if (workers == 0) workers = std::max(1u, std::thread::hardware_concurrency());
    for (unsigned i = 0; i < workers; ++i) threads_.emplace_back([this] { worker_loop(); });
}

JobRunner::~JobRunner() {
    {
        std::lock_guard lock(mu_);
        stop_ = true;
    }
    work_cv_.notify_all();
    for (auto& t : threads_) t.join();
}

std::future<void> JobRunner::submit(std::string name, std::function<void()> fn, int max_attempts) {
    auto promise = std::make_shared<std::promise<void>>();
    auto fut = promise->get_future();
    {
        std::lock_guard lock(mu_);
        queue_.push_back({std::move(name), std::move(fn), std::max(1, max_attempts), std::move(promise)});
        ++stats_.submitted;
        stats_.max_queue_depth = std::max(stats_.max_queue_depth, queue_.size());
    }
    work_cv_.notify_one();
    return fut;
}

void JobRunner::wait_idle() {
    std::unique_lock lock(mu_);
    idle_cv_.wait(lock, [&] { return queue_.empty() && running_ == 0; });
}

JobRunner::Stats JobRunner::stats() const {
    std::lock_guard lock(mu_);
    return stats_;
}

void JobRunner::worker_loop() {
    for (;;) {
        Job job;
        {
            std::unique_lock lock(mu_);
            work_cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
            if (queue_.empty()) return;  // stop_ and drained
            job = std::move(queue_.front());
            queue_.pop_front();
            ++running_;
        }

        const auto t0 = std::chrono::steady_clock::now();
        uint64_t retries = 0;
        std::exception_ptr last_error;
        bool ok = false;
        for (int attempt = 1; attempt <= job.max_attempts && !ok; ++attempt) {
            try {
                job.fn();
                ok = true;
            } catch (...) {
                last_error = std::current_exception();
                if (attempt < job.max_attempts) ++retries;
            }
        }
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        if (ok)
            job.promise->set_value();
        else
            job.promise->set_exception(last_error);

        {
            std::lock_guard lock(mu_);
            --running_;
            stats_.busy_seconds += secs;
            stats_.retries += retries;
            ok ? ++stats_.completed : ++stats_.failed;
            if (queue_.empty() && running_ == 0) idle_cv_.notify_all();
        }
    }
}

}  // namespace ff
