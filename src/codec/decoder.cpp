#include "codec/decoder.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

namespace ff {
namespace {

[[noreturn]] void av_fail(const std::string& what, int err) {
    char buf[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror(err, buf, sizeof buf);
    throw std::runtime_error(what + ": " + buf);
}

// Copy a decoded AVFrame into an I420 Frame, via swscale when the format or size differs.
void convert_to_i420(const AVFrame* src, Frame& dst, SwsContext*& sws) {
    const bool direct = (src->format == AV_PIX_FMT_YUV420P) && src->width == dst.width() && src->height == dst.height();
    if (direct) {
        for (int p = 0; p < 3; ++p)
            for (int r = 0; r < dst.plane_height(p); ++r)
                std::memcpy(dst.data(p) + r * dst.stride(p), src->data[p] + static_cast<ptrdiff_t>(r) * src->linesize[p],
                            static_cast<size_t>(dst.plane_width(p)));
        return;
    }
    sws = sws_getCachedContext(sws, src->width, src->height, static_cast<AVPixelFormat>(src->format), dst.width(),
                               dst.height(), AV_PIX_FMT_YUV420P, SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!sws) throw std::runtime_error("sws_getCachedContext failed");
    uint8_t* d[4] = {dst.data(0), dst.data(1), dst.data(2), nullptr};
    int ls[4] = {static_cast<int>(dst.stride(0)), static_cast<int>(dst.stride(1)), static_cast<int>(dst.stride(2)), 0};
    sws_scale(sws, src->data, src->linesize, 0, src->height, d, ls);
}

}  // namespace

// ------------------------------------------------------------------ Decoder

struct Decoder::Impl {
    AVFormatContext* fmt = nullptr;
    AVCodecContext* ctx = nullptr;
    AVPacket* pkt = nullptr;
    AVFrame* frame = nullptr;
    SwsContext* sws = nullptr;
    int stream = -1;
    bool eof_sent = false;
    int64_t index = 0;
    VideoInfo info;

    ~Impl() {
        sws_freeContext(sws);
        av_frame_free(&frame);
        av_packet_free(&pkt);
        avcodec_free_context(&ctx);
        avformat_close_input(&fmt);
    }
};

Decoder::Decoder() : impl_(std::make_unique<Impl>()) {}
Decoder::~Decoder() = default;
const VideoInfo& Decoder::info() const noexcept { return impl_->info; }

std::unique_ptr<Decoder> Decoder::open(const std::string& path) {
    std::unique_ptr<Decoder> d(new Decoder());
    Impl& m = *d->impl_;

    int ret = avformat_open_input(&m.fmt, path.c_str(), nullptr, nullptr);
    if (ret < 0) av_fail("cannot open " + path, ret);
    ret = avformat_find_stream_info(m.fmt, nullptr);
    if (ret < 0) av_fail("cannot read stream info", ret);

    const AVCodec* codec = nullptr;
    m.stream = av_find_best_stream(m.fmt, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
    if (m.stream < 0) throw std::runtime_error("no video stream in " + path);
    AVStream* st = m.fmt->streams[m.stream];

    m.ctx = avcodec_alloc_context3(codec);
    if (!m.ctx) throw std::bad_alloc();
    ret = avcodec_parameters_to_context(m.ctx, st->codecpar);
    if (ret < 0) av_fail("avcodec_parameters_to_context", ret);
    m.ctx->thread_count = 0;  // let libavcodec pick
    ret = avcodec_open2(m.ctx, codec, nullptr);
    if (ret < 0) av_fail("avcodec_open2", ret);

    m.pkt = av_packet_alloc();
    m.frame = av_frame_alloc();
    if (!m.pkt || !m.frame) throw std::bad_alloc();

    const AVRational fr = av_guess_frame_rate(m.fmt, st, nullptr);
    m.info.width = m.ctx->width & ~1;
    m.info.height = m.ctx->height & ~1;
    m.info.fps = (fr.num > 0 && fr.den > 0) ? av_q2d(fr) : 30.0;
    if (m.fmt->duration != AV_NOPTS_VALUE) m.info.duration_s = static_cast<double>(m.fmt->duration) / AV_TIME_BASE;
    m.info.frame_count_hint = st->nb_frames;
    m.info.codec_name = codec->name;
    return d;
}

FramePtr Decoder::next(const std::shared_ptr<FramePool>& pool) {
    Impl& m = *impl_;
    for (;;) {
        int ret = avcodec_receive_frame(m.ctx, m.frame);
        if (ret == 0) {
            FramePtr out = pool->acquire();
            if (!out) return nullptr;
            convert_to_i420(m.frame, *out, m.sws);
            out->pts = m.index++;
            av_frame_unref(m.frame);
            return out;
        }
        if (ret == AVERROR_EOF) return nullptr;
        if (ret != AVERROR(EAGAIN)) av_fail("avcodec_receive_frame", ret);

        if (m.eof_sent) return nullptr;
        ret = av_read_frame(m.fmt, m.pkt);
        if (ret < 0) {  // end of file (or read error): drain the decoder
            avcodec_send_packet(m.ctx, nullptr);
            m.eof_sent = true;
            continue;
        }
        if (m.pkt->stream_index == m.stream) {
            ret = avcodec_send_packet(m.ctx, m.pkt);
            if (ret < 0 && ret != AVERROR(EAGAIN)) {
                av_packet_unref(m.pkt);
                av_fail("avcodec_send_packet", ret);
            }
        }
        av_packet_unref(m.pkt);
    }
}

// ------------------------------------------------------------- PacketDecoder

struct PacketDecoder::Impl {
    AVCodecContext* ctx = nullptr;
    AVCodecParserContext* parser = nullptr;
    AVPacket* pkt = nullptr;
    AVFrame* frame = nullptr;
    SwsContext* sws = nullptr;
    std::unique_ptr<Frame> out;
    int64_t index = 0;

    ~Impl() {
        sws_freeContext(sws);
        av_frame_free(&frame);
        av_packet_free(&pkt);
        if (parser) av_parser_close(parser);
        avcodec_free_context(&ctx);
    }

    void drain(const FrameSink& sink) {
        for (;;) {
            const int ret = avcodec_receive_frame(ctx, frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) return;
            if (ret < 0) av_fail("avcodec_receive_frame", ret);
            const int w = frame->width & ~1, h = frame->height & ~1;
            if (!out || out->width() != w || out->height() != h) out = std::make_unique<Frame>(w, h);
            convert_to_i420(frame, *out, sws);
            out->pts = index++;
            av_frame_unref(frame);
            sink(*out);
        }
    }
};

PacketDecoder::PacketDecoder(const std::string& codec_name) : impl_(std::make_unique<Impl>()) {
    const AVCodec* codec = avcodec_find_decoder_by_name(codec_name.c_str());
    if (!codec) throw std::runtime_error("no decoder named " + codec_name);
    Impl& m = *impl_;
    m.ctx = avcodec_alloc_context3(codec);
    if (!m.ctx) throw std::bad_alloc();
    m.ctx->thread_count = 1;
    const int ret = avcodec_open2(m.ctx, codec, nullptr);
    if (ret < 0) av_fail("avcodec_open2", ret);
    m.pkt = av_packet_alloc();
    m.frame = av_frame_alloc();
    if (!m.pkt || !m.frame) throw std::bad_alloc();
}

PacketDecoder::~PacketDecoder() = default;

void PacketDecoder::decode(const uint8_t* data, size_t size, const FrameSink& sink) {
    Impl& m = *impl_;
    int ret = av_new_packet(m.pkt, static_cast<int>(size));  // allocates the required input padding
    if (ret < 0) av_fail("av_new_packet", ret);
    std::memcpy(m.pkt->data, data, size);
    ret = avcodec_send_packet(m.ctx, m.pkt);
    av_packet_unref(m.pkt);
    if (ret < 0 && ret != AVERROR(EAGAIN)) av_fail("avcodec_send_packet", ret);
    m.drain(sink);
}

void PacketDecoder::decode_stream(const uint8_t* data, size_t size, const FrameSink& sink) {
    Impl& m = *impl_;
    if (!m.parser) {
        m.parser = av_parser_init(m.ctx->codec_id);
        if (!m.parser) throw std::runtime_error("no parser for codec");
    }
    while (size > 0) {
        uint8_t* pd = nullptr;
        int ps = 0;
        const int used = av_parser_parse2(m.parser, m.ctx, &pd, &ps, data, static_cast<int>(size), AV_NOPTS_VALUE,
                                          AV_NOPTS_VALUE, 0);
        if (used < 0) av_fail("av_parser_parse2", used);
        data += used;
        size -= static_cast<size_t>(used);
        if (ps > 0) decode(pd, static_cast<size_t>(ps), sink);
    }
    uint8_t* pd = nullptr;
    int ps = 0;
    av_parser_parse2(m.parser, m.ctx, &pd, &ps, nullptr, 0, AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);  // flush last AU
    if (ps > 0) decode(pd, static_cast<size_t>(ps), sink);
}

void PacketDecoder::flush(const FrameSink& sink) {
    Impl& m = *impl_;
    avcodec_send_packet(m.ctx, nullptr);
    m.drain(sink);
}

}  // namespace ff
