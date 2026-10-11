#pragma once
// libav-backed encoding (libx264 / libx265 by default). No libav types here.

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "pipeline/frame.hpp"

namespace ff {

struct EncodeParams {
    std::string codec = "libx264";  // libx264 or libx265
    std::string preset = "veryfast";
    int crf = 23;
    int width = 0;
    int height = 0;
    double fps = 30.0;
    int keyint = 0;     // max GOP length; 0 = encoder default
    int threads = 1;    // threads inside one encode (parallelism normally comes from running many jobs)
    int bframes = -1;   // -1 = encoder default
    std::string extra;  // codec private params, "key=value:key=value"
};

struct PacketView {
    const uint8_t* data;
    size_t size;
    int64_t pts;
    bool keyframe;
};
using PacketSink = std::function<void(const PacketView&)>;

class Encoder {
public:
    static std::unique_ptr<Encoder> create(const EncodeParams& params);  // throws on failure
    ~Encoder();
    Encoder(const Encoder&) = delete;
    Encoder& operator=(const Encoder&) = delete;

    // Frames must have the configured dimensions. Packets are delivered through
    // `sink` as soon as the encoder emits them (it may buffer for lookahead / B-frames).
    void encode(const Frame& frame, const PacketSink& sink);
    void flush(const PacketSink& sink);

private:
    Encoder();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// "libx264" -> "h264", "libx265" -> "hevc"
std::string decoder_name_for(const std::string& encoder_name);

// Silence libav / x264 / x265 console logging (errors only).
void set_codec_log_quiet();

struct RoundTripResult {
    size_t bytes = 0;
    size_t frames = 0;
    std::vector<uint8_t> bitstream;  // filled only if keep_bitstream
};

// Encodes `frames`, decodes every packet straight back, and calls
// on_recon(index, reconstructed_frame) in display order. This is how rate and
// quality are measured without touching the disk.
RoundTripResult encode_roundtrip(const std::vector<FramePtr>& frames, const EncodeParams& params,
                                 const std::function<void(size_t, const Frame&)>& on_recon,
                                 bool keep_bitstream = false);

}  // namespace ff
