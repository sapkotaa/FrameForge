// Needs the libav codec layer. Exercises encode -> decode round trips, the
// parser against real x264 output, chunked encoding and the probe.

#include <filesystem>
#include <fstream>

#include "codec/decoder.hpp"
#include "codec/encoder.hpp"
#include "codec/h264_parser.hpp"
#include "optimizer/chunked.hpp"
#include "optimizer/probe.hpp"
#include "quality/psnr.hpp"
#include "test_util.hpp"

using namespace ff;

namespace {
EncodeParams params(int w, int h, int crf) {
    EncodeParams p;
    p.width = w;
    p.height = h;
    p.crf = crf;
    p.fps = 30;
    p.preset = "ultrafast";
    return p;
}

double mean_psnr(const std::vector<FramePtr>& src, const EncodeParams& p, size_t* bytes = nullptr,
                 std::vector<uint8_t>* stream = nullptr) {
    double sum = 0;
    auto rt = encode_roundtrip(src, p, [&](size_t i, const Frame& f) { sum += psnr_y(*src[i], f); }, stream != nullptr);
    if (bytes) *bytes = rt.bytes;
    if (stream) *stream = std::move(rt.bitstream);
    return sum / static_cast<double>(src.size());
}

std::string temp_path(const char* name) { return (std::filesystem::temp_directory_path() / name).string(); }
}  // namespace

TEST(roundtrip_returns_every_frame_with_good_quality) {
    set_codec_log_quiet();
    const auto clip = fftest::make_clip(128, 96, 20);
    size_t bytes = 0;
    const double q = mean_psnr(clip, params(128, 96, 20), &bytes);
    CHECK(bytes > 0);
    CHECK(q > 33.0);
}

TEST(lower_crf_means_more_bits_and_better_quality) {
    const auto clip = fftest::make_clip(128, 96, 20);
    size_t b_hi = 0, b_lo = 0;
    const double q_hi = mean_psnr(clip, params(128, 96, 18), &b_hi);
    const double q_lo = mean_psnr(clip, params(128, 96, 38), &b_lo);
    CHECK(b_hi > b_lo);
    CHECK(q_hi > q_lo);
}

TEST(parser_reads_real_x264_output) {
    const auto clip = fftest::make_clip(130, 98, 24);  // 130x98 forces cropping: coded 144x112
    EncodeParams p = params(130, 98, 23);
    p.keyint = 12;
    p.preset = "veryfast";
    std::vector<uint8_t> stream;
    mean_psnr(clip, p, nullptr, &stream);
    const auto st = h264::analyze_annexb(stream.data(), stream.size());
    REQUIRE(st.sps.has_value());
    CHECK_EQ(st.sps->width, 130);
    CHECK_EQ(st.sps->height, 98);
    CHECK_EQ(st.sps->coded_width, 144);
    CHECK_EQ(st.sps->coded_height, 112);
    CHECK_EQ(st.sps->profile_idc, 100);  // x264 default: High
    REQUIRE(st.pps.has_value());
    CHECK(st.pps->cabac);
    CHECK_EQ(st.frames.size(), size_t(24));
    CHECK(st.frames.front().idr);
    CHECK(st.idr_frames >= 2);  // keyint 12 over 24 frames
    for (size_t g : st.gop_lengths) CHECK(g <= 12);
    // An I-frame costs more than the average P-frame on this content.
    double ib = 0, pb = 0;
    size_t ni = 0, np = 0;
    for (const auto& f : st.frames) {
        if (f.type == 'I') { ib += double(f.bytes); ++ni; }
        if (f.type == 'P') { pb += double(f.bytes); ++np; }
    }
    if (ni && np) CHECK(ib / double(ni) > pb / double(np));
    CHECK(st.sps->timing_present);
    CHECK_NEAR(st.sps->fps(), 30.0, 0.01);
}

TEST(packet_decoder_decodes_a_raw_annexb_stream) {
    const auto clip = fftest::make_clip(128, 96, 16);
    std::vector<uint8_t> stream;
    mean_psnr(clip, params(128, 96, 23), nullptr, &stream);
    PacketDecoder dec("h264");
    size_t n = 0;
    auto count = [&](const Frame& f) {
        CHECK_EQ(f.width(), 128);
        ++n;
    };
    dec.decode_stream(stream.data(), stream.size(), count);
    dec.flush(count);
    CHECK_EQ(n, size_t(16));
}

TEST(file_decoder_reads_what_the_encoder_wrote) {
    const auto clip = fftest::make_clip(128, 96, 18);
    std::vector<uint8_t> stream;
    mean_psnr(clip, params(128, 96, 23), nullptr, &stream);
    const std::string path = temp_path("frameforge_test_clip.h264");
    {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(stream.data()), static_cast<std::streamsize>(stream.size()));
    }
    auto dec = Decoder::open(path);
    CHECK_EQ(dec->info().width, 128);
    CHECK_EQ(dec->info().height, 96);
    auto pool = make_frame_pool(128, 96, 32);
    size_t n = 0;
    while (FramePtr f = dec->next(pool)) {
        CHECK_EQ(f->pts, int64_t(n));
        ++n;
    }
    CHECK_EQ(n, size_t(18));

    // And through load_clip(): decode -> scene-cut -> collect pipeline.
    const SourceClip sc = load_clip(path, 10);
    CHECK_EQ(sc.frames.size(), size_t(10));
    CHECK_EQ(sc.load_stats.stages.size(), size_t(3));
    std::filesystem::remove(path);
}

TEST(decoder_reports_a_missing_file) {
    bool threw = false;
    try {
        Decoder::open("/definitely/not/here.mp4");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

TEST(encoder_rejects_bad_input) {
    EncodeParams p = params(128, 96, 23);
    p.codec = "not-a-codec";
    bool threw = false;
    try {
        Encoder::create(p);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
    p = params(127, 96, 23);
    threw = false;
    try {
        Encoder::create(p);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

TEST(chunked_encode_is_a_valid_deterministic_stream) {
    const auto clip = fftest::make_clip(128, 96, 30);
    const auto plan = plan_chunks(clip.size(), {}, 8, 10);  // 10,10,10
    REQUIRE(plan.size() == 3);
    JobRunner runner(3);

    EncodeParams p = params(128, 96, 23);
    const ChunkedResult a = encode_chunked(clip, p, plan, runner);
    const ChunkedResult b = encode_chunked(clip, p, plan, runner);
    CHECK(a.bitstream == b.bitstream);  // same inputs, same bytes: the jobs are idempotent
    CHECK_EQ(a.chunks.size(), size_t(3));

    const auto st = h264::analyze_annexb(a.bitstream.data(), a.bitstream.size());
    CHECK_EQ(st.frames.size(), size_t(30));
    CHECK_EQ(st.idr_frames, size_t(3));  // one IDR per chunk
    CHECK_EQ(st.nal_counts[h264::kNalSps], size_t(3));  // headers repeated, so each chunk stands alone
    CHECK_EQ(st.gop_lengths.size(), size_t(2));
    for (size_t g : st.gop_lengths) CHECK_EQ(g, size_t(10));

    // Decode the stitched stream end to end and compare with the source.
    PacketDecoder dec("h264");
    size_t n = 0;
    double sum = 0;
    auto check = [&](const Frame& f) {
        if (n < clip.size()) sum += psnr_y(*clip[n], f);
        ++n;
    };
    dec.decode_stream(a.bitstream.data(), a.bitstream.size(), check);
    dec.flush(check);
    CHECK_EQ(n, size_t(30));
    CHECK(sum / 30.0 > 33.0);
}

TEST(chunk_size_does_not_change_quality_much_but_costs_bits) {
    const auto clip = fftest::make_clip(128, 96, 48);
    JobRunner runner(2);
    EncodeParams p = params(128, 96, 23);
    p.preset = "veryfast";
    const auto one = encode_chunked(clip, p, plan_chunks(48, {}, 48, 48), runner);
    const auto many = encode_chunked(clip, p, plan_chunks(48, {}, 8, 8), runner);
    CHECK(many.bitstream.size() > one.bitstream.size());  // more IDR frames and repeated headers
    CHECK(many.bitstream.size() < one.bitstream.size() * 2);
}

TEST(probe_produces_a_sensible_rate_quality_grid) {
    SourceClip clip;
    clip.width = 128;
    clip.height = 96;
    clip.fps = 30;
    clip.frames = fftest::make_clip(128, 96, 16);

    ProbeConfig cfg;
    cfg.preset = "ultrafast";
    cfg.heights = {96, 48};
    cfg.crfs = {20, 36};
    JobRunner runner(4);
    const auto pts = probe_rate_quality(clip, cfg, runner);
    REQUIRE(pts.size() == 4);
    auto at = [&](int h, int crf) {
        for (const auto& p : pts)
            if (p.height == h && p.crf == crf) return p;
        throw std::runtime_error("missing probe point");
    };
    CHECK(at(96, 20).kbps > at(96, 36).kbps);
    CHECK(at(96, 20).psnr > at(96, 36).psnr);
    CHECK(at(48, 20).kbps < at(96, 20).kbps);    // fewer pixels, fewer bits
    CHECK(at(48, 20).psnr < at(96, 20).psnr);    // ...but measured against the full-size source it looks worse
    CHECK(at(96, 20).ssim > 0.5 && at(96, 20).ssim <= 1.0);
    CHECK_EQ(at(48, 20).width, 64);
}

TEST(ladder_heights_and_widths) {
    const auto h = default_ladder_heights(1080);
    CHECK_EQ(h.front(), 1080);
    CHECK(std::find(h.begin(), h.end(), 720) != h.end());
    CHECK(std::find(h.begin(), h.end(), 1440) == h.end());
    CHECK_EQ(even_width_for_height(1920, 1080, 720), 1280);
    CHECK_EQ(even_width_for_height(1920, 1080, 360), 640);
    CHECK_EQ(even_width_for_height(1280, 720, 540) % 2, 0);
    CHECK_EQ(default_ladder_heights(100).front(), 100);
}
