#pragma once
// Split-and-stitch encoding.
//
// The clip is cut into chunks (preferably at scene cuts, never longer than
// max_len). Each chunk is encoded independently on its own thread with the same
// quality settings (constant CRF keeps rate control consistent across chunks),
// then the chunk bitstreams are concatenated. Every chunk starts with an IDR
// frame preceded by fresh SPS/PPS and no frame references data across a chunk
// boundary (closed GOP), so the concatenation is a valid stream.

#include <cstdint>
#include <vector>

#include "codec/encoder.hpp"
#include "optimizer/job_runner.hpp"
#include "pipeline/frame.hpp"

namespace ff {

struct Chunk {
    size_t start = 0;
    size_t count = 0;
};

// Cuts [0, n_frames) into chunks. Boundaries prefer `scene_cuts` (frame indices
// where a new scene starts) once a chunk has min_len frames; a chunk is forced
// to end at max_len. Chunks are contiguous and cover every frame exactly once;
// every chunk except possibly the last has at least min_len frames.
std::vector<Chunk> plan_chunks(size_t n_frames, const std::vector<int>& scene_cuts, size_t min_len, size_t max_len);

struct ChunkResult {
    Chunk chunk;
    size_t bytes = 0;
    double seconds = 0;
};

struct ChunkedResult {
    std::vector<uint8_t> bitstream;  // Annex B, chunks concatenated in order
    std::vector<ChunkResult> chunks;
    double wall_seconds = 0;
    double cpu_seconds = 0;  // sum of per-chunk encode times
    double speedup() const { return wall_seconds > 0 ? cpu_seconds / wall_seconds : 1.0; }
};






#if defined(FF_WITH_FFMPEG)
ChunkedResult encode_chunked(const std::vector<FramePtr>& frames, const EncodeParams& params,
                             const std::vector<Chunk>& plan, JobRunner& runner);
#endif





}  // namespace ff
