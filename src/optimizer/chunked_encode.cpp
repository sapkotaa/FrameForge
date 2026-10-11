#include <chrono>
#include <future>

#include "optimizer/chunked.hpp"

namespace ff {

ChunkedResult encode_chunked(const std::vector<FramePtr>& frames, const EncodeParams& params,
                             const std::vector<Chunk>& plan, JobRunner& runner) {
    using Clock = std::chrono::steady_clock;
    ChunkedResult res;
    res.chunks.resize(plan.size());
    std::vector<std::vector<uint8_t>> parts(plan.size());

    const auto t0 = Clock::now();
    std::vector<std::future<void>> futures;
    for (size_t i = 0; i < plan.size(); ++i) {
        futures.push_back(runner.submit("chunk " + std::to_string(i), [&, i] {
            // Idempotent: the job owns parts[i] / res.chunks[i] outright and rebuilds
            // them from scratch, so a retry after a failure can't leave partial output.
            const auto c0 = Clock::now();
            parts[i].clear();
            const Chunk ch = plan[i];

            EncodeParams p = params;
            p.keyint = static_cast<int>(ch.count);  // one GOP per chunk, starting on an IDR
            auto enc = Encoder::create(p);
            auto sink = [&](const PacketView& pv) { parts[i].insert(parts[i].end(), pv.data, pv.data + pv.size); };
            for (size_t k = 0; k < ch.count; ++k) enc->encode(*frames[ch.start + k], sink);
            enc->flush(sink);

            res.chunks[i].chunk = ch;
            res.chunks[i].bytes = parts[i].size();
            res.chunks[i].seconds = std::chrono::duration<double>(Clock::now() - c0).count();
        }));
    }
    for (auto& f : futures) f.get();  // rethrows if a chunk failed all of its attempts
    res.wall_seconds = std::chrono::duration<double>(Clock::now() - t0).count();

    for (size_t i = 0; i < plan.size(); ++i) {
        res.bitstream.insert(res.bitstream.end(), parts[i].begin(), parts[i].end());
        res.cpu_seconds += res.chunks[i].seconds;
    }
    return res;
}

}  // namespace ff
