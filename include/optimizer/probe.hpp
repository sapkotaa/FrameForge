#pragma once
// Loading a clip and probing its rate-quality behaviour.

#include <string>
#include <vector>

#include "optimizer/job_runner.hpp"
#include "optimizer/ladder.hpp"
#include "pipeline/frame.hpp"
#include "pipeline/pipeline.hpp"

namespace ff {

struct SourceClip {
    std::string path;
    int width = 0, height = 0;
    double fps = 30;
    std::vector<FramePtr> frames;
    std::vector<int> scene_cuts;   // frame indices where a new scene starts
    PipelineStats load_stats;      // decode -> scene-cut -> collect pipeline counters
};

// Decodes up to max_frames (<= 0: the whole file) through a three-stage
// pipeline: decode, scene-cut detection, collect. All frames stay in memory.
SourceClip load_clip(const std::string& path, int max_frames);

// Heights to probe: the standard streaming ladder steps not above the source,
// plus the source height itself.
std::vector<int> default_ladder_heights(int source_height);
int even_width_for_height(int src_w, int src_h, int target_h);

struct ProbeConfig {
    std::string codec = "libx264";
    std::string preset = "veryfast";
    std::vector<int> heights;
    std::vector<int> crfs = {18, 22, 26, 30, 34, 38};
    bool measure_ssim = true;
};




// Encodes the clip at every (height, crf) pair as parallel jobs. Each
// reconstruction is upscaled back to the source size and compared with the
// source, so rungs at different resolutions land on one quality axis.
std::vector<RatePoint> probe_rate_quality(const SourceClip& clip, const ProbeConfig& cfg, JobRunner& runner);



}  // namespace ff
