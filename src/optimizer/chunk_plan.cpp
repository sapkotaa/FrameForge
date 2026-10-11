#include <algorithm>
#include <stdexcept>

#include "optimizer/chunked.hpp"

namespace ff {

std::vector<Chunk> plan_chunks(size_t n_frames, const std::vector<int>& scene_cuts, size_t min_len, size_t max_len) {
    if (min_len == 0 || max_len < min_len) throw std::invalid_argument("plan_chunks: need 0 < min_len <= max_len");
    std::vector<Chunk> out;
    size_t start = 0;
    size_t ci = 0;
    while (start < n_frames) {
        const size_t hard_end = std::min(n_frames, start + max_len);
        size_t end = hard_end;
        // First scene cut that leaves at least min_len frames in this chunk.
        while (ci < scene_cuts.size() && static_cast<size_t>(scene_cuts[ci]) < start + min_len) ++ci;
        if (ci < scene_cuts.size() && static_cast<size_t>(scene_cuts[ci]) < hard_end) end = static_cast<size_t>(scene_cuts[ci]);
        out.push_back({start, end - start});
        start = end;
    }
    return out;
}

}  // namespace ff
