#include "optimizer/probe.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <map>
#include <stdexcept>

#include "codec/decoder.hpp"
#include "codec/encoder.hpp"
#include "kernels/resize.hpp"
#include "pipeline/scene_cut.hpp"
#include "quality/psnr.hpp"
#include "quality/ssim.hpp"

namespace ff {

SourceClip load_clip(const std::string& path, int max_frames) {
    auto dec = Decoder::open(path);
    const VideoInfo& vi = dec->info();

    SourceClip clip;
    clip.path = path;
    clip.width = vi.width;
    clip.height = vi.height;
    clip.fps = vi.fps;

    size_t want = max_frames > 0 ? static_cast<size_t>(max_frames)
                                 : static_cast<size_t>(std::max<int64_t>(vi.frame_count_hint, static_cast<int64_t>(vi.duration_s * vi.fps) + 1));
    if (want == 0) want = 1;
    auto pool = make_frame_pool(vi.width, vi.height, want);

    int taken = 0;
    SceneCutDetector detector;
    FramePipeline pipe(4);
    pipe.add_stage("scene-cut", [&](FramePtr f) {
        if (detector.push(*f)) clip.scene_cuts.push_back(static_cast<int>(f->pts));
        return f;
    });
    clip.load_stats = pipe.run(
        [&]() -> FramePtr {
            if (max_frames > 0 && taken >= max_frames) return nullptr;
            if (pool->available() == 0) return nullptr;  // pool sized for the whole request
            FramePtr f = dec->next(pool);
            if (f) ++taken;
            return f;
        },
        [&](FramePtr f) { clip.frames.push_back(std::move(f)); },
        [&] { pool->close(); });
    return clip;
}

std::vector<int> default_ladder_heights(int source_height) {
    static const int kSteps[] = {2160, 1440, 1080, 720, 540, 360, 240};
    std::vector<int> out;
    for (int h : kSteps)
        if (h < source_height) out.push_back(h);
    out.insert(out.begin(), source_height);
    return out;
}

int even_width_for_height(int src_w, int src_h, int target_h) {
    const int w = static_cast<int>(std::lround(static_cast<double>(src_w) * target_h / src_h));
    return std::max(2, (w + 1) & ~1);
}

std::vector<RatePoint> probe_rate_quality(const SourceClip& clip, const ProbeConfig& cfg, JobRunner& runner) {
    if (clip.frames.empty()) throw std::invalid_argument("probe: clip has no frames");
    const size_t n = clip.frames.size();

    // Phase 1: build the scaled copies of the source, one job per height.
    std::map<int, std::vector<FramePtr>> scaled;
    std::vector<std::future<void>> futs;
    for (int h : cfg.heights) {
        auto& slot = scaled[h];
        if (h == clip.height) {
            slot = clip.frames;  // shared, not copied
            continue;
        }
        slot.resize(n);
        const int w = even_width_for_height(clip.width, clip.height, h);
        futs.push_back(runner.submit("scale " + std::to_string(h), [&clip, &slot, w, h, n] {
            for (size_t i = 0; i < n; ++i) {
                auto f = std::make_shared<Frame>(w, h);
                resize_frame(*clip.frames[i], *f);
                slot[i] = std::move(f);
            }
        }));
    }
    for (auto& f : futs) f.get();
    futs.clear();

    // Phase 2: one job per (height, crf).
    std::vector<RatePoint> points(cfg.heights.size() * cfg.crfs.size());
    const double video_seconds = static_cast<double>(n) / clip.fps;
    size_t slot_index = 0;
    for (int h : cfg.heights) {
        for (int crf : cfg.crfs) {
            RatePoint& pt = points[slot_index++];
            const auto& frames = scaled.at(h);
            futs.push_back(runner.submit("probe " + std::to_string(h) + "p crf" + std::to_string(crf), [&, h, crf] {
                const auto t0 = std::chrono::steady_clock::now();
                const int w = frames.front()->width();

                EncodeParams ep;
                ep.codec = cfg.codec;
                ep.preset = cfg.preset;
                ep.crf = crf;
                ep.width = w;
                ep.height = h;
                ep.fps = clip.fps;

                Frame upscaled(clip.width, clip.height);
                double psnr_sum = 0, ssim_sum = 0;
                const RoundTripResult rt = encode_roundtrip(frames, ep, [&](size_t i, const Frame& rec) {
                    const Frame* cmp = &rec;
                    if (rec.width() != clip.width || rec.height() != clip.height) {
                        resize_frame(rec, upscaled);
                        cmp = &upscaled;
                    }
                    psnr_sum += psnr_y(*clip.frames[i], *cmp);
                    if (cfg.measure_ssim) ssim_sum += ssim_y(*clip.frames[i], *cmp);
                });

                pt = RatePoint{};  // overwrite the whole slot: safe to retry
                pt.width = w;
                pt.height = h;
                pt.crf = crf;
                pt.kbps = static_cast<double>(rt.bytes) * 8.0 / 1000.0 / video_seconds;
                pt.psnr = psnr_sum / static_cast<double>(n);
                pt.ssim = cfg.measure_ssim ? ssim_sum / static_cast<double>(n) : 0.0;
                pt.cpu_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            }));
        }
    }
    for (auto& f : futs) f.get();
    return points;
}

}  // namespace ff
