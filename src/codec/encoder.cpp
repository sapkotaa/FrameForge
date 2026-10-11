#include "codec/encoder.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

#include "codec/decoder.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavutil/log.h>
#include <libavutil/opt.h>
#include <libavutil/rational.h>
}

namespace ff {
namespace {
[[noreturn]] void av_fail(const std::string& what, int err) {
    char buf[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror(err, buf, sizeof buf);
    throw std::runtime_error(what + ": " + buf);
}
}  // namespace

struct Encoder::Impl {
    AVCodecContext* ctx = nullptr;
    AVFrame* frame = nullptr;
    AVPacket* pkt = nullptr;
    int64_t next_pts = 0;
    int width = 0, height = 0;

    ~Impl() {
        av_packet_free(&pkt);
        av_frame_free(&frame);
        avcodec_free_context(&ctx);
    }

    void drain(const PacketSink& sink) {
        for (;;) {
            const int ret = avcodec_receive_packet(ctx, pkt);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) return;
            if (ret < 0) av_fail("avcodec_receive_packet", ret);
            sink(PacketView{pkt->data, static_cast<size_t>(pkt->size), pkt->pts, (pkt->flags & AV_PKT_FLAG_KEY) != 0});
            av_packet_unref(pkt);
        }
    }
};

Encoder::Encoder() : impl_(std::make_unique<Impl>()) {}
Encoder::~Encoder() = default;

std::unique_ptr<Encoder> Encoder::create(const EncodeParams& p) {
    if (p.width <= 0 || p.height <= 0 || p.width % 2 || p.height % 2)
        throw std::invalid_argument("Encoder: dimensions must be positive and even");
    const AVCodec* codec = avcodec_find_encoder_by_name(p.codec.c_str());
    if (!codec) throw std::runtime_error("encoder not available in this libav build: " + p.codec);

    std::unique_ptr<Encoder> e(new Encoder());
    Impl& m = *e->impl_;
    m.width = p.width;
    m.height = p.height;
    m.ctx = avcodec_alloc_context3(codec);
    if (!m.ctx) throw std::bad_alloc();

    const AVRational fr = av_d2q(p.fps > 0 ? p.fps : 30.0, 100000);
    m.ctx->width = p.width;
    m.ctx->height = p.height;
    m.ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    m.ctx->framerate = fr;
    m.ctx->time_base = av_inv_q(fr);
    m.ctx->thread_count = p.threads > 0 ? p.threads : 1;
    if (p.keyint > 0) m.ctx->gop_size = p.keyint;
    if (p.bframes >= 0) m.ctx->max_b_frames = p.bframes;

    AVDictionary* opts = nullptr;
    av_dict_set(&opts, "preset", p.preset.c_str(), 0);
    av_dict_set(&opts, "crf", std::to_string(p.crf).c_str(), 0);
    if (!p.extra.empty()) {
        const char* key = (p.codec == "libx265") ? "x265-params" : "x264-params";
        av_dict_set(&opts, key, p.extra.c_str(), 0);
    }
    const int ret = avcodec_open2(m.ctx, codec, &opts);
    av_dict_free(&opts);
    if (ret < 0) av_fail("avcodec_open2(" + p.codec + ")", ret);

    m.frame = av_frame_alloc();
    m.pkt = av_packet_alloc();
    if (!m.frame || !m.pkt) throw std::bad_alloc();
    m.frame->format = AV_PIX_FMT_YUV420P;
    m.frame->width = p.width;
    m.frame->height = p.height;
    if (av_frame_get_buffer(m.frame, 0) < 0) throw std::bad_alloc();
    return e;
}

void Encoder::encode(const Frame& f, const PacketSink& sink) {
    Impl& m = *impl_;
    if (f.width() != m.width || f.height() != m.height) throw std::invalid_argument("Encoder::encode: frame size mismatch");
    int ret = av_frame_make_writable(m.frame);
    if (ret < 0) av_fail("av_frame_make_writable", ret);
    for (int p = 0; p < 3; ++p)
        for (int r = 0; r < f.plane_height(p); ++r)
            std::memcpy(m.frame->data[p] + static_cast<ptrdiff_t>(r) * m.frame->linesize[p], f.data(p) + r * f.stride(p),
                        static_cast<size_t>(f.plane_width(p)));
    m.frame->pts = m.next_pts++;
    ret = avcodec_send_frame(m.ctx, m.frame);
    if (ret < 0) av_fail("avcodec_send_frame", ret);
    m.drain(sink);
}

void Encoder::flush(const PacketSink& sink) {
    Impl& m = *impl_;
    const int ret = avcodec_send_frame(m.ctx, nullptr);
    if (ret < 0 && ret != AVERROR_EOF) av_fail("avcodec_send_frame(flush)", ret);
    m.drain(sink);
}

std::string decoder_name_for(const std::string& enc) {
    if (enc == "libx264") return "h264";
    if (enc == "libx265") return "hevc";
    throw std::invalid_argument("no known decoder for encoder " + enc);
}

void set_codec_log_quiet() { av_log_set_level(AV_LOG_ERROR); }

RoundTripResult encode_roundtrip(const std::vector<FramePtr>& frames, const EncodeParams& params,
                                 const std::function<void(size_t, const Frame&)>& on_recon, bool keep_bitstream) {
    RoundTripResult res;
    auto enc = Encoder::create(params);
    PacketDecoder dec(decoder_name_for(params.codec));

    size_t recon_index = 0;
    auto on_frame = [&](const Frame& f) {
        if (on_recon) on_recon(recon_index, f);
        ++recon_index;
    };
    auto on_packet = [&](const PacketView& pv) {
        res.bytes += pv.size;
        if (keep_bitstream) res.bitstream.insert(res.bitstream.end(), pv.data, pv.data + pv.size);
        dec.decode(pv.data, pv.size, on_frame);
    };
    for (const auto& f : frames) enc->encode(*f, on_packet);
    enc->flush(on_packet);
    dec.flush(on_frame);

    res.frames = recon_index;
    if (recon_index != frames.size())
        throw std::runtime_error("round trip produced " + std::to_string(recon_index) + " frames, expected " +
                                 std::to_string(frames.size()));
    return res;
}

}  // namespace ff
