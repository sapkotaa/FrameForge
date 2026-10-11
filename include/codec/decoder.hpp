#pragma once
// libav-backed decoding. No libav types leak into this header.

#include <functional>
#include <memory>
#include <string>

#include "pipeline/frame.hpp"

namespace ff {

struct VideoInfo {
    int width = 0;   // rounded down to even (I420 needs even dimensions)
    int height = 0;
    double fps = 0;
    double duration_s = 0;
    int64_t frame_count_hint = 0;  // from the container, 0 if unknown
    std::string codec_name;
};

// Demux + decode a video file, converting every frame to 8-bit I420.
class Decoder {
public:
    // Throws std::runtime_error with the libav error text on failure.
    static std::unique_ptr<Decoder> open(const std::string& path);
    ~Decoder();
    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;

    const VideoInfo& info() const noexcept;

    // Next frame in presentation order, taken from `pool` (blocks while the
    // pool is exhausted). nullptr at end of stream. pts is the frame index.
    FramePtr next(const std::shared_ptr<FramePool>& pool);

private:
    Decoder();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Decodes bare codec packets (no container), e.g. straight from an encoder.
class PacketDecoder {
public:
    using FrameSink = std::function<void(const Frame&)>;  // frame is only valid during the call

    explicit PacketDecoder(const std::string& codec_name);  // "h264" or "hevc"
    ~PacketDecoder();
    PacketDecoder(const PacketDecoder&) = delete;
    PacketDecoder& operator=(const PacketDecoder&) = delete;

    // One complete access unit.
    void decode(const uint8_t* data, size_t size, const FrameSink& sink);
    // A raw Annex B byte stream: split into access units with libav's parser.
    void decode_stream(const uint8_t* data, size_t size, const FrameSink& sink);
    void flush(const FrameSink& sink);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ff
