#pragma once
// A small thread-pool job runner for the transcoding "DAG": probe jobs, chunk
// encodes, etc. Jobs are retried on failure, so a job must be idempotent: it
// has to write its result by overwriting a fixed slot, never by appending, so
// a retry (or a duplicate run) can't corrupt the output. That is the same
// contract a distributed scheduler needs for preemptible workers.

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ff {

class JobRunner {
public:
    struct Stats {
        uint64_t submitted = 0;
        uint64_t completed = 0;
        uint64_t failed = 0;          // jobs that exhausted their attempts
        uint64_t retries = 0;         // extra attempts across all jobs
        double busy_seconds = 0;      // summed wall time workers spent inside jobs
        size_t max_queue_depth = 0;
    };

    explicit JobRunner(unsigned workers = 0);  // 0 = hardware_concurrency
    ~JobRunner();                              // finishes queued work, then joins
    JobRunner(const JobRunner&) = delete;
    JobRunner& operator=(const JobRunner&) = delete;

    // The future carries the last exception if every attempt failed.
    std::future<void> submit(std::string name, std::function<void()> fn, int max_attempts = 3);

    void wait_idle();  // returns when the queue is empty and no job is running
    Stats stats() const;
    unsigned workers() const noexcept { return static_cast<unsigned>(threads_.size()); }

private:
    struct Job {
        std::string name;
        std::function<void()> fn;
        int max_attempts;
        std::shared_ptr<std::promise<void>> promise;
    };
    void worker_loop();

    mutable std::mutex mu_;
    std::condition_variable work_cv_;
    std::condition_variable idle_cv_;
    std::deque<Job> queue_;
    size_t running_ = 0;
    bool stop_ = false;
    Stats stats_;
    std::vector<std::thread> threads_;
};

}  // namespace ff
