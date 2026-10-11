#pragma once
// A small threaded frame-processing pipeline: source -> stage* -> sink.
//
// Every stage runs on its own thread and neighbours are connected by bounded
// SPSC channels, so a slow stage throttles everything upstream instead of
// letting queues (and memory) grow. Per-stage counters and timers make the
// bottleneck stage obvious.

#include <exception>
#include <functional>
#include <string>
#include <vector>

#include "pipeline/frame.hpp"
#include "pipeline/ring_buffer.hpp"

namespace ff {

struct StageStats {
    std::string name;
    uint64_t frames_in = 0;
    uint64_t frames_out = 0;
    double busy_seconds = 0;     // time spent inside the stage's own function
    uint64_t in_stalls = 0;      // times the stage found its input queue empty
    uint64_t out_stalls = 0;     // times the stage found its output queue full
};

struct PipelineStats {
    double wall_seconds = 0;
    std::vector<StageStats> stages;  // source, transforms..., sink
};

class FramePipeline {
public:
    using Source = std::function<FramePtr()>;             // nullptr = end of stream
    using Transform = std::function<FramePtr(FramePtr)>;  // return nullptr to drop the frame
    using Sink = std::function<void(FramePtr)>;

    explicit FramePipeline(size_t queue_depth = 4) : depth_(queue_depth) {}

    FramePipeline& add_stage(std::string name, Transform fn) {
        stages_.push_back({std::move(name), std::move(fn)});
        return *this;
    }

    // Blocks until the stream is drained. The sink runs on the calling thread.
    // If any callback throws, every queue is cancelled and the first exception
    // is rethrown here after all threads have joined.
    //
    // A source that can block on something other than the queues (typically
    // FramePool::acquire() when every frame is in flight) must be woken when
    // the pipeline is torn down, or run() would wait for it forever. Pass
    // `on_cancel` for that, e.g. `[&] { pool->close(); }`; it is called once,
    // from the failing thread, after the queues are cancelled.
    PipelineStats run(Source source, Sink sink, std::function<void()> on_cancel = {});

private:
    struct Stage {
        std::string name;
        Transform fn;
    };
    size_t depth_;
    std::vector<Stage> stages_;
};

}  // namespace ff
