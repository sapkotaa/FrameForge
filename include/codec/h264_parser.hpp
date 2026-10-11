#pragma once
// Minimal H.264 (AVC) Annex B bitstream parser: NAL splitting, emulation
// prevention removal, SPS/PPS/slice-header decoding, and stream statistics.
// Written from the spec (ITU-T H.264 section 7), no libav involved.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ff::h264 {

enum NalType : int {
    kNalSlice = 1,
    kNalSliceIdr = 5,
    kNalSei = 6,
    kNalSps = 7,
    kNalPps = 8,
    kNalAud = 9,
    kNalEndSeq = 10,
    kNalEndStream = 11,
    kNalFiller = 12,
};

struct NalUnit {
    int ref_idc = 0;      // nal_ref_idc: 0 = non-reference
    int type = 0;         // nal_unit_type
    const uint8_t* data;  // points at the NAL header byte inside the caller's buffer
    size_t size = 0;      // including the header byte, excluding the start code
    size_t offset = 0;    // offset of this NAL's start code within the buffer (for size accounting)
};

// Splits an Annex B byte stream on 00 00 01 / 00 00 00 01 start codes.
// Trailing zero bytes (trailing_zero_8bits) are stripped from each NAL.
std::vector<NalUnit> split_annexb(const uint8_t* data, size_t size);

// Removes emulation prevention bytes (00 00 03 -> 00 00) to give the RBSP.
std::vector<uint8_t> unescape_rbsp(const uint8_t* payload, size_t size);

struct Sps {
    int profile_idc = 0;
    int constraint_flags = 0;  // constraint_set0..5 + reserved bits, as the raw byte
    int level_idc = 0;
    int sps_id = 0;
    int chroma_format_idc = 1;  // 1 = 4:2:0
    int bit_depth_luma = 8;
    int bit_depth_chroma = 8;
    int log2_max_frame_num = 4;
    int pic_order_cnt_type = 0;
    int max_num_ref_frames = 0;
    bool frame_mbs_only = true;
    int coded_width = 0;   // macroblock-aligned
    int coded_height = 0;
    int width = 0;         // after cropping, e.g. 1088 -> 1080
    int height = 0;
    // VUI (optional): -1 when absent
    int sar_width = -1, sar_height = -1;
    int video_full_range = -1;
    int colour_primaries = -1, transfer = -1, matrix = -1;
    bool timing_present = false;
    uint32_t num_units_in_tick = 0, time_scale = 0;
    double fps() const {  // H.264 timing is per field: fps = time_scale / (2 * num_units_in_tick)
        return (timing_present && num_units_in_tick) ? time_scale / (2.0 * num_units_in_tick) : 0.0;
    }
};

struct Pps {
    int pps_id = 0;
    int sps_id = 0;
    bool cabac = false;  // entropy_coding_mode_flag: CABAC vs CAVLC
    int num_slice_groups = 1;
    int num_ref_idx_l0_default = 1;
    int num_ref_idx_l1_default = 1;
    bool weighted_pred = false;
    int weighted_bipred_idc = 0;
    int pic_init_qp = 26;
    bool transform_8x8_mode = false;
};

enum class SliceType { P = 0, B = 1, I = 2, SP = 3, SI = 4 };

struct SliceHeader {
    uint32_t first_mb = 0;
    SliceType type = SliceType::I;
};

// `nal` includes the one-byte NAL header. Return nullopt on malformed/truncated data.
std::optional<Sps> parse_sps(const uint8_t* nal, size_t size);
std::optional<Pps> parse_pps(const uint8_t* nal, size_t size);
std::optional<SliceHeader> parse_slice_header(const uint8_t* nal, size_t size);

const char* profile_name(int profile_idc);
std::string level_name(int level_idc, int constraint_flags, int profile_idc);
const char* colour_primaries_name(int v);

struct FrameInfo {
    char type = '?';       // 'I', 'P', 'B'
    bool idr = false;
    size_t bytes = 0;      // all NAL bytes (incl. start codes) attributed to this frame
};

struct StreamStats {
    size_t total_bytes = 0;
    size_t nal_counts[32] = {};
    std::optional<Sps> sps;
    std::optional<Pps> pps;
    std::vector<FrameInfo> frames;  // decode order
    size_t idr_frames = 0, i_frames = 0, p_frames = 0, b_frames = 0;
    std::vector<size_t> gop_lengths;  // frames between consecutive IDR frames
};

StreamStats analyze_annexb(const uint8_t* data, size_t size);

}  // namespace ff::h264
