// frameforge ladder clip.mp4      per-title bitrate ladder (probe -> Pareto frontier / hull -> rungs)
// frameforge compare clip.mp4     BD-rate between two encoders / presets
// frameforge analyze stream.h264  inspect an Annex B H.264 stream (SPS, GOP structure)
// frameforge cpu                  which SIMD backend is active

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "codec/h264_parser.hpp"
#include "kernels/dispatch.hpp"
#include "kernels/yuv.hpp"
#include "optimizer/chunked.hpp"
#include "optimizer/ladder.hpp"
#include "quality/bdrate.hpp"

#if defined(FF_WITH_FFMPEG)
#include "codec/decoder.hpp"
#include "codec/encoder.hpp"
#include "kernels/resize.hpp"
#include "optimizer/probe.hpp"
#include "quality/psnr.hpp"
#endif

namespace {

using namespace ff;

void usage() {
    std::puts(
        "FrameForge - media processing framework and per-title ladder optimizer\n"
        "\n"
        "usage:\n"
        "  frameforge ladder <clip> [options]    build a bitrate ladder for a clip\n"
        "      --frames N        frames to analyze (default 90, 0 = whole clip)\n"
        "      --crfs a,b,c      CRF values to probe (default 18,22,26,30,34,38)\n"
        "      --heights a,b,c   rendition heights (default: standard steps up to the source)\n"
        "      --codec NAME      libx264 (default) or libx265\n"
        "      --preset NAME     encoder preset (default veryfast)\n"
        "      --jobs N          worker threads (default: all cores)\n"
        "      --max-rungs N     ladder size cap (default 6)\n"
        "      --out DIR         write probes.csv, ladder.json (and rung streams with --encode)\n"
        "      --encode          chunk-parallel encode of every rung into --out, then verify\n"
        "      --chunk-min N / --chunk-max N   chunk length bounds in frames (default 24 / 48)\n"
        "      --price X         USD per CPU-hour for the cost line (default 0.04)\n"
        "  frameforge compare <clip> [--a-codec libx264 --a-preset medium --b-codec libx265 --b-preset medium]\n"
        "                            [--frames N] [--crfs a,b,c,d] [--jobs N]\n"
        "  frameforge analyze <file.h264>        inspect an Annex B H.264 elementary stream\n"
        "  frameforge cpu                        show SIMD backends and the active one");
}

#if defined(FF_WITH_FFMPEG)
std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) out.push_back(std::stoi(tok));
    return out;
}
#endif

struct Args {
    std::vector<std::string> positional;
    std::vector<std::pair<std::string, std::string>> opts;

    static bool is_flag(const std::string& k) { return k == "--encode" || k == "--help"; }

    explicit Args(int argc, char** argv) {
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if (a.rfind("--", 0) == 0) {
                if (is_flag(a)) {
                    opts.emplace_back(a, "1");
                } else if (i + 1 < argc) {
                    opts.emplace_back(a, argv[++i]);
                } else {
                    throw std::runtime_error("missing value for " + a);
                }
            } else {
                positional.push_back(a);
            }
        }
    }
    std::string get(const std::string& k, const std::string& def = "") const {
        for (const auto& [key, v] : opts)
            if (key == k) return v;
        return def;
    }
    bool has(const std::string& k) const {
        for (const auto& [key, v] : opts)
            if (key == k) return true;
        return false;
    }
};

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), {});
}

#if defined(FF_WITH_FFMPEG)
void write_file(const std::string& path, const uint8_t* data, size_t n) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path);
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n));
}
#endif

// ----------------------------------------------------------------------- cpu
int cmd_cpu() {
#if defined(__aarch64__)
    const char* arch = "arm64";
#elif defined(__x86_64__)
    const char* arch = "x86_64";
#else
    const char* arch = "other";
#endif
    std::printf("arch:            %s\n", arch);
    std::printf("hardware threads: %u\n", std::thread::hardware_concurrency());
    std::printf("simd backends:   ");
    for (Isa i : supported_isas()) std::printf("%s ", to_string(i));
    std::printf("\nactive backend:  %s  (override with FRAMEFORGE_ISA=scalar|neon|avx2)\n", to_string(kernels().isa));
#if defined(FF_WITH_FFMPEG)
    std::puts("libav codec layer: enabled");
#else
    std::puts("libav codec layer: disabled (build with -DFRAMEFORGE_WITH_FFMPEG=ON)");
#endif
    return 0;
}

// ------------------------------------------------------------------- analyze
int cmd_analyze(const Args& a) {
    if (a.positional.empty()) {
        usage();
        return 2;
    }
    const auto bytes = read_file(a.positional[0]);
    const h264::StreamStats st = h264::analyze_annexb(bytes.data(), bytes.size());
    if (!st.sps) {
        std::fprintf(stderr,
                     "no SPS found: %s is not an Annex B H.264 stream.\n"
                     "extract one from an MP4 with:\n"
                     "  ffmpeg -i in.mp4 -an -c:v copy -bsf:v h264_mp4toannexb out.h264\n",
                     a.positional[0].c_str());
        return 1;
    }
    const h264::Sps& s = *st.sps;
    std::printf("stream:        %s (%zu bytes)\n", a.positional[0].c_str(), st.total_bytes);
    std::printf("profile/level: %s, level %s  (profile_idc %d)\n", h264::profile_name(s.profile_idc),
                h264::level_name(s.level_idc, s.constraint_flags, s.profile_idc).c_str(), s.profile_idc);
    std::printf("picture:       %dx%d (coded %dx%d), chroma_format_idc %d (%s), %d-bit\n", s.width, s.height, s.coded_width,
                s.coded_height, s.chroma_format_idc, s.chroma_format_idc == 1 ? "4:2:0" : "other", s.bit_depth_luma);
    std::printf("refs/poc:      max_num_ref_frames %d, poc_type %d, frame_mbs_only %d\n", s.max_num_ref_frames, s.pic_order_cnt_type,
                s.frame_mbs_only);
    if (st.pps) std::printf("entropy:       %s, transform_8x8 %s, weighted_pred %d\n", st.pps->cabac ? "CABAC" : "CAVLC",
                            st.pps->transform_8x8_mode ? "on" : "off", st.pps->weighted_pred);
    if (s.colour_primaries >= 0)
        std::printf("colour:        primaries %s, range %s\n", h264::colour_primaries_name(s.colour_primaries),
                    s.video_full_range == 1 ? "full" : "limited");
    const double fps = s.fps();
    if (fps > 0) std::printf("timing (VUI):  %.3f fps\n", fps);

    std::printf("NAL units:     SPS %zu  PPS %zu  SEI %zu  IDR slices %zu  non-IDR slices %zu\n", st.nal_counts[h264::kNalSps],
                st.nal_counts[h264::kNalPps], st.nal_counts[h264::kNalSei], st.nal_counts[h264::kNalSliceIdr],
                st.nal_counts[h264::kNalSlice]);
    std::printf("frames:        %zu  (I %zu, P %zu, B %zu), %zu IDR\n", st.frames.size(), st.i_frames, st.p_frames, st.b_frames,
                st.idr_frames);
    if (!st.gop_lengths.empty()) {
        size_t mx = *std::max_element(st.gop_lengths.begin(), st.gop_lengths.end());
        double avg = 0;
        for (size_t g : st.gop_lengths) avg += static_cast<double>(g);
        avg /= static_cast<double>(st.gop_lengths.size());
        std::printf("GOP length:    avg %.1f, max %zu frames\n", avg, mx);
    }
    double sum[256] = {};
    size_t cnt[256] = {};
    for (const auto& f : st.frames) {
        sum[static_cast<unsigned char>(f.type)] += static_cast<double>(f.bytes);
        ++cnt[static_cast<unsigned char>(f.type)];
    }
    for (char t : {'I', 'P', 'B'})
        if (cnt[static_cast<unsigned char>(t)])
            std::printf("avg %c-frame:   %.0f bytes\n", t, sum[static_cast<unsigned char>(t)] / static_cast<double>(cnt[static_cast<unsigned char>(t)]));
    if (fps > 0 && !st.frames.empty())
        std::printf("avg bitrate:   %.0f kbps\n", static_cast<double>(st.total_bytes) * 8.0 / 1000.0 / (static_cast<double>(st.frames.size()) / fps));
    std::string pattern;
    for (size_t i = 0; i < st.frames.size() && i < 72; ++i) pattern += st.frames[i].idr ? 'R' : st.frames[i].type;
    std::printf("decode order:  %s%s   (R = IDR)\n", pattern.c_str(), st.frames.size() > 72 ? "..." : "");
    return 0;
}

#if defined(FF_WITH_FFMPEG)

// -------------------------------------------------------------------- ladder
std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') o += '\\';
        o += c;
    }
    return o;
}

int cmd_ladder(const Args& a) {
    if (a.positional.empty()) {
        usage();
        return 2;
    }
    set_codec_log_quiet();
    const std::string path = a.positional[0];
    const int max_frames = std::stoi(a.get("--frames", "90"));
    const unsigned jobs = static_cast<unsigned>(std::stoi(a.get("--jobs", "0")));
    const double price = std::stod(a.get("--price", "0.04"));

    std::fprintf(stderr, "loading %s ...\n", path.c_str());
    const SourceClip clip = load_clip(path, max_frames);
    if (clip.frames.empty()) {
        std::fprintf(stderr, "no frames decoded\n");
        return 1;
    }
    const double video_s = static_cast<double>(clip.frames.size()) / clip.fps;
    std::printf("FrameForge ladder\n");
    std::printf("  input    %s: %dx%d @ %.2f fps, %zu frames (%.1f s), %zu scene cut(s)\n", path.c_str(), clip.width, clip.height,
                clip.fps, clip.frames.size(), video_s, clip.scene_cuts.size());
    if (!clip.scene_cuts.empty()) {
        std::printf("  cuts at ");
        for (int c : clip.scene_cuts) std::printf("frame %d (%.2f s)  ", c, c / clip.fps);
        std::printf("\n");
    }
    std::printf("  raw      %.0f Mbps uncompressed 8-bit 4:2:0\n", raw_bitrate_mbps(clip.width, clip.height, clip.fps));

    JobRunner runner(jobs);
    std::printf("  runtime  simd=%s, %u worker threads\n", to_string(kernels().isa), runner.workers());

    ProbeConfig pc;
    pc.codec = a.get("--codec", "libx264");
    pc.preset = a.get("--preset", "veryfast");
    pc.crfs = parse_int_list(a.get("--crfs", "18,22,26,30,34,38"));
    pc.heights = a.has("--heights") ? parse_int_list(a.get("--heights")) : default_ladder_heights(clip.height);

    std::printf("  probing  %zu renditions x %zu CRFs = %zu encodes (%s, preset %s)\n", pc.heights.size(), pc.crfs.size(),
                pc.heights.size() * pc.crfs.size(), pc.codec.c_str(), pc.preset.c_str());
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<RatePoint> points = probe_rate_quality(clip, pc, runner);
    const double probe_wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    const std::vector<RatePoint> frontier = pareto_frontier(points);
    const std::vector<RatePoint> hull = convex_hull(points);
    LadderConfig lc;
    lc.max_rungs = std::stoi(a.get("--max-rungs", "6"));
    const std::vector<RatePoint> ladder = select_ladder(points, lc);

    auto in_set = [](const std::vector<RatePoint>& v, const RatePoint& p) {
        return std::any_of(v.begin(), v.end(), [&](const RatePoint& q) { return q.height == p.height && q.crf == p.crf; });
    };

    std::sort(points.begin(), points.end(), [](const RatePoint& x, const RatePoint& y) { return x.kbps < y.kbps; });
    std::printf("\n  %-9s %-4s %9s %8s %7s   %s\n", "size", "crf", "kbps", "PSNR dB", "SSIM", "");
    for (const auto& p : points)
        std::printf("  %4dx%-4d %-4d %9.0f %8.2f %7.4f   %s\n", p.width, p.height, p.crf, p.kbps, p.psnr, p.ssim,
                    in_set(ladder, p) ? "<- rung" : (in_set(hull, p) ? "hull" : (in_set(frontier, p) ? "frontier" : "")));

    std::printf("\n  bitrate ladder (%zu rungs; %zu probes -> %zu on the Pareto frontier -> %zu on the convex hull)\n", ladder.size(),
                points.size(), frontier.size(), hull.size());
    for (size_t i = 0; i < ladder.size(); ++i) {
        const RatePoint& r = ladder[i];
        std::printf("    #%zu  %4dx%-4d  crf %-2d  %7.0f kbps   PSNR %.2f dB   SSIM %.4f\n", i + 1, r.width, r.height, r.crf, r.kbps,
                    r.psnr, r.ssim);
    }

    const JobRunner::Stats js = runner.stats();
    double cpu = 0;
    for (const auto& p : points) cpu += p.cpu_seconds;
    std::printf("\n  probe wall %.1f s, cpu %.1f s (%.1fx parallel), %llu jobs, %llu retries\n", probe_wall, cpu, cpu / probe_wall,
                static_cast<unsigned long long>(js.completed), static_cast<unsigned long long>(js.retries));
    std::printf("  analysis cost: $%.4f per video minute at $%.2f/CPU-hour\n", cost_per_video_minute(cpu, video_s, price), price);
    if (!ladder.empty()) {
        // Illustrative break-even: the top rung vs a fixed 1080p-style 5 Mbps rung would be a policy decision;
        // report the raw numbers a capacity planner needs instead.
        const double gb_per_hour = ladder.back().kbps * 1000.0 / 8.0 * 3600.0 / 1e9;
        std::printf("  top rung streams %.2f GB per viewed hour\n", gb_per_hour);
    }

    // ---- outputs
    const std::string out_dir = a.get("--out");
    if (!out_dir.empty()) {
        std::system(("mkdir -p '" + out_dir + "'").c_str());
        std::ofstream csv(out_dir + "/probes.csv");
        csv << "width,height,crf,kbps,psnr,ssim,cpu_seconds,on_frontier,on_hull,on_ladder\n";
        for (const auto& p : points)
            csv << p.width << ',' << p.height << ',' << p.crf << ',' << p.kbps << ',' << p.psnr << ',' << p.ssim << ',' << p.cpu_seconds
                << ',' << in_set(frontier, p) << ',' << in_set(hull, p) << ',' << in_set(ladder, p) << '\n';
        std::ofstream js_out(out_dir + "/ladder.json");
        js_out << "{\n  \"source\": \"" << json_escape(path) << "\",\n  \"width\": " << clip.width << ", \"height\": " << clip.height
               << ", \"fps\": " << clip.fps << ", \"frames\": " << clip.frames.size() << ",\n  \"codec\": \"" << pc.codec
               << "\", \"preset\": \"" << pc.preset << "\",\n  \"rungs\": [\n";
        for (size_t i = 0; i < ladder.size(); ++i) {
            const RatePoint& r = ladder[i];
            js_out << "    {\"width\": " << r.width << ", \"height\": " << r.height << ", \"crf\": " << r.crf
                   << ", \"kbps\": " << std::llround(r.kbps) << ", \"psnr\": " << r.psnr << ", \"ssim\": " << r.ssim << "}"
                   << (i + 1 < ladder.size() ? "," : "") << "\n";
        }
        js_out << "  ]\n}\n";
        std::printf("  wrote %s/probes.csv and %s/ladder.json\n", out_dir.c_str(), out_dir.c_str());
    }

    if (a.has("--encode")) {
        if (out_dir.empty()) {
            std::fprintf(stderr, "--encode needs --out DIR\n");
            return 2;
        }
        const size_t cmin = static_cast<size_t>(std::stoi(a.get("--chunk-min", "24")));
        const size_t cmax = static_cast<size_t>(std::stoi(a.get("--chunk-max", "48")));
        const auto plan = plan_chunks(clip.frames.size(), clip.scene_cuts, cmin, cmax);
        std::printf("\n  chunked encode: %zu chunks (%zu..%zu frames, split at scene cuts when possible)\n", plan.size(), cmin, cmax);
        for (const auto& r : ladder) {
            std::vector<FramePtr> frames;
            if (r.height == clip.height) {
                frames = clip.frames;
            } else {
                for (const auto& f : clip.frames) {
                    auto s = std::make_shared<Frame>(r.width, r.height);
                    resize_frame(*f, *s);
                    frames.push_back(std::move(s));
                }
            }
            EncodeParams ep;
            ep.codec = pc.codec;
            ep.preset = pc.preset;
            ep.crf = r.crf;
            ep.width = r.width;
            ep.height = r.height;
            ep.fps = clip.fps;
            const ChunkedResult cr = encode_chunked(frames, ep, plan, runner);
            const std::string file = out_dir + "/rung_" + std::to_string(r.height) + "p_crf" + std::to_string(r.crf) + ".h264";
            write_file(file, cr.bitstream.data(), cr.bitstream.size());

            // Verify the stitched stream: decode it whole and compare with the source rendition.
            size_t decoded = 0;
            double psnr_sum = 0;
            PacketDecoder dec(decoder_name_for(pc.codec));
            auto check = [&](const Frame& f) {
                if (decoded < frames.size()) psnr_sum += psnr_y(*frames[decoded], f);
                ++decoded;
            };
            dec.decode_stream(cr.bitstream.data(), cr.bitstream.size(), check);
            dec.flush(check);
            const double kbps = static_cast<double>(cr.bitstream.size()) * 8.0 / 1000.0 / video_s;
            std::printf("    %4dp  %7.0f kbps  wall %.2fs  cpu %.2fs  (%.1fx)  decoded %zu/%zu frames, PSNR vs source rendition %.2f dB  -> %s\n",
                        r.height, kbps, cr.wall_seconds, cr.cpu_seconds, cr.speedup(), decoded, frames.size(),
                        decoded ? psnr_sum / static_cast<double>(decoded) : 0.0, file.c_str());
            if (decoded != frames.size()) {
                std::fprintf(stderr, "    VERIFY FAILED: decoded %zu of %zu frames\n", decoded, frames.size());
                return 1;
            }
        }
    }
    return 0;
}

// ------------------------------------------------------------------- compare
int cmd_compare(const Args& a) {
    if (a.positional.empty()) {
        usage();
        return 2;
    }
    set_codec_log_quiet();
    const SourceClip clip = load_clip(a.positional[0], std::stoi(a.get("--frames", "60")));
    if (clip.frames.empty()) return 1;
    JobRunner runner(static_cast<unsigned>(std::stoi(a.get("--jobs", "0"))));

    ProbeConfig base;
    base.heights = {clip.height};
    base.crfs = parse_int_list(a.get("--crfs", "20,25,30,35"));
    ProbeConfig ca = base, cb = base;
    ca.codec = a.get("--a-codec", "libx264");
    ca.preset = a.get("--a-preset", "medium");
    cb.codec = a.get("--b-codec", "libx265");
    cb.preset = a.get("--b-preset", "medium");

    auto run = [&](const ProbeConfig& pc) {
        std::vector<RdPoint> rd;
        std::printf("  %s %s\n", pc.codec.c_str(), pc.preset.c_str());
        for (const auto& p : probe_rate_quality(clip, pc, runner)) {
            std::printf("    crf %-3d %9.0f kbps  PSNR %.2f dB\n", p.crf, p.kbps, p.psnr);
            rd.push_back({p.kbps, p.psnr});
        }
        return rd;
    };
    std::printf("FrameForge compare: %dx%d, %zu frames\n", clip.width, clip.height, clip.frames.size());
    const auto A = run(ca);
    const auto B = run(cb);
    const double bd = bd_rate(A, B);
    std::printf("\n  BD-rate of B vs A (PSNR): %+.1f%%  (negative = B needs fewer bits at equal quality)\n", bd);
    return 0;
}

#endif  // FF_WITH_FFMPEG

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "help") == 0) {
        usage();
        return argc < 2 ? 2 : 0;
    }
    const std::string cmd = argv[1];
    try {
        const Args args(argc, argv);
        if (cmd == "cpu") return cmd_cpu();
        if (cmd == "analyze") return cmd_analyze(args);
#if defined(FF_WITH_FFMPEG)
        if (cmd == "ladder") return cmd_ladder(args);
        if (cmd == "compare") return cmd_compare(args);
#else
        if (cmd == "ladder" || cmd == "compare") {
            std::fprintf(stderr, "this build has no libav support; reconfigure with -DFRAMEFORGE_WITH_FFMPEG=ON\n");
            return 2;
        }
#endif
        usage();
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
