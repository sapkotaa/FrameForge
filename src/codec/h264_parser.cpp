#include "codec/h264_parser.hpp"

#include <algorithm>
#include <stdexcept>

namespace ff::h264 {
namespace {

struct ParseError : std::runtime_error {
    ParseError() : std::runtime_error("h264: truncated or malformed bitstream") {}
};

class BitReader {
public:
    BitReader(const uint8_t* d, size_t n) : d_(d), nbits_(n * 8) {}

    uint32_t bit() {
        if (pos_ >= nbits_) throw ParseError();
        const uint32_t b = (d_[pos_ >> 3] >> (7 - (pos_ & 7))) & 1u;
        ++pos_;
        return b;
    }
    uint32_t u(int n) {  // n <= 32
        uint32_t v = 0;
        for (int i = 0; i < n; ++i) v = (v << 1) | bit();
        return v;
    }
    bool flag() { return bit() != 0; }
    // Exp-Golomb, ue(v): leading zeros, a one, then that many bits.
    uint32_t ue() {
        int zeros = 0;
        while (bit() == 0) {
            if (++zeros > 31) throw ParseError();
        }
        if (zeros == 0) return 0;
        return ((1u << zeros) - 1u) + u(zeros);
    }
    int32_t se() {  // se(v): 0, 1, -1, 2, -2, ...
        const uint32_t k = ue();
        const int32_t mag = static_cast<int32_t>((k + 1) >> 1);
        return (k & 1) ? mag : -mag;
    }
    size_t pos() const { return pos_; }
    // True if payload bits remain before the rbsp_trailing_bits (the final 1 bit + zero padding).
    bool more_rbsp_data() const {
        size_t last = nbits_;
        while (last > 0) {
            const size_t byte = (last - 1) >> 3;
            if (d_[byte] == 0) {
                last = byte * 8;
                continue;
            }
            // position (from start) of the last set bit
            int bit_in_byte = 7;
            while (((d_[byte] >> (7 - bit_in_byte)) & 1) == 0) --bit_in_byte;
            last = byte * 8 + static_cast<size_t>(bit_in_byte);
            return pos_ < last;
        }
        return false;
    }

private:
    const uint8_t* d_;
    size_t nbits_;
    size_t pos_ = 0;
};

void skip_scaling_list(BitReader& br, int size) {
    int last = 8, next = 8;
    for (int j = 0; j < size; ++j) {
        if (next != 0) {
            const int delta = br.se();
            next = (last + delta + 256) % 256;
        }
        last = (next == 0) ? last : next;
    }
}

bool is_high_profile(int p) {
    switch (p) {
        case 100: case 110: case 122: case 244: case 44: case 83: case 86:
        case 118: case 128: case 138: case 139: case 134: case 135:
            return true;
        default:
            return false;
    }
}

void parse_vui(BitReader& br, Sps& s) {
    if (br.flag()) {  // aspect_ratio_info_present_flag
        const uint32_t idc = br.u(8);
        if (idc == 255) {
            s.sar_width = static_cast<int>(br.u(16));
            s.sar_height = static_cast<int>(br.u(16));
        }
    }
    if (br.flag()) br.flag();  // overscan_info_present_flag -> overscan_appropriate_flag
    if (br.flag()) {           // video_signal_type_present_flag
        br.u(3);               // video_format
        s.video_full_range = br.flag() ? 1 : 0;
        if (br.flag()) {       // colour_description_present_flag
            s.colour_primaries = static_cast<int>(br.u(8));
            s.transfer = static_cast<int>(br.u(8));
            s.matrix = static_cast<int>(br.u(8));
        }
    }
    if (br.flag()) {  // chroma_loc_info_present_flag
        br.ue();
        br.ue();
    }
    if (br.flag()) {  // timing_info_present_flag
        s.num_units_in_tick = br.u(32);
        s.time_scale = br.u(32);
        s.timing_present = true;
        // fixed_frame_rate_flag and the HRD parameters follow; not needed here.
    }
}

}  // namespace

std::vector<NalUnit> split_annexb(const uint8_t* d, size_t n) {
    std::vector<NalUnit> out;
    size_t i = 0;
    auto find_start = [&](size_t from) -> size_t {  // index of the first byte of a 00 00 01 pattern, or n
        for (size_t k = from; k + 2 < n; ++k)
            if (d[k] == 0 && d[k + 1] == 0 && d[k + 2] == 1) return k;
        return n;
    };
    i = find_start(0);
    while (i < n) {
        const size_t nal_begin = i + 3;
        const size_t next = find_start(nal_begin);
        size_t nal_end = next;  // strip zero_byte of a following 4-byte start code and trailing zeros
        while (nal_end > nal_begin && d[nal_end - 1] == 0) --nal_end;
        if (nal_end > nal_begin) {
            NalUnit u;
            u.data = d + nal_begin;
            u.size = nal_end - nal_begin;
            u.offset = (i > 0 && d[i - 1] == 0) ? i - 1 : i;  // include the zero_byte of a 4-byte start code
            u.ref_idc = (d[nal_begin] >> 5) & 3;
            u.type = d[nal_begin] & 31;
            out.push_back(u);
        }
        i = next;
    }
    return out;
}

std::vector<uint8_t> unescape_rbsp(const uint8_t* p, size_t n) {
    std::vector<uint8_t> out;
    out.reserve(n);
    int zeros = 0;
    for (size_t i = 0; i < n; ++i) {
        if (zeros >= 2 && p[i] == 3) {  // emulation_prevention_three_byte
            zeros = 0;
            continue;
        }
        out.push_back(p[i]);
        zeros = (p[i] == 0) ? zeros + 1 : 0;
    }
    return out;
}

std::optional<Sps> parse_sps(const uint8_t* nal, size_t size) {
    if (size < 5 || (nal[0] & 31) != kNalSps) return std::nullopt;
    const std::vector<uint8_t> rbsp = unescape_rbsp(nal + 1, size - 1);
    try {
        BitReader br(rbsp.data(), rbsp.size());
        Sps s;
        s.profile_idc = static_cast<int>(br.u(8));
        s.constraint_flags = static_cast<int>(br.u(8));
        s.level_idc = static_cast<int>(br.u(8));
        s.sps_id = static_cast<int>(br.ue());
        bool separate_colour_plane = false;
        if (is_high_profile(s.profile_idc)) {
            s.chroma_format_idc = static_cast<int>(br.ue());
            if (s.chroma_format_idc == 3) separate_colour_plane = br.flag();
            s.bit_depth_luma = 8 + static_cast<int>(br.ue());
            s.bit_depth_chroma = 8 + static_cast<int>(br.ue());
            br.flag();  // qpprime_y_zero_transform_bypass_flag
            if (br.flag()) {  // seq_scaling_matrix_present_flag
                const int lists = (s.chroma_format_idc != 3) ? 8 : 12;
                for (int i = 0; i < lists; ++i)
                    if (br.flag()) skip_scaling_list(br, i < 6 ? 16 : 64);
            }
        }
        s.log2_max_frame_num = 4 + static_cast<int>(br.ue());
        s.pic_order_cnt_type = static_cast<int>(br.ue());
        if (s.pic_order_cnt_type == 0) {
            br.ue();  // log2_max_pic_order_cnt_lsb_minus4
        } else if (s.pic_order_cnt_type == 1) {
            br.flag();
            br.se();
            br.se();
            const uint32_t n = br.ue();
            if (n > 255) return std::nullopt;
            for (uint32_t i = 0; i < n; ++i) br.se();
        }
        s.max_num_ref_frames = static_cast<int>(br.ue());
        br.flag();  // gaps_in_frame_num_value_allowed_flag
        const int mbs_w = static_cast<int>(br.ue()) + 1;
        const int map_units_h = static_cast<int>(br.ue()) + 1;
        s.frame_mbs_only = br.flag();
        if (!s.frame_mbs_only) br.flag();  // mb_adaptive_frame_field_flag
        br.flag();                         // direct_8x8_inference_flag
        int crop_l = 0, crop_r = 0, crop_t = 0, crop_b = 0;
        if (br.flag()) {  // frame_cropping_flag
            crop_l = static_cast<int>(br.ue());
            crop_r = static_cast<int>(br.ue());
            crop_t = static_cast<int>(br.ue());
            crop_b = static_cast<int>(br.ue());
        }
        const int chroma_array_type = separate_colour_plane ? 0 : s.chroma_format_idc;
        const int sub_w = (chroma_array_type == 1 || chroma_array_type == 2) ? 2 : 1;
        const int sub_h = (chroma_array_type == 1) ? 2 : 1;
        const int crop_unit_x = chroma_array_type == 0 ? 1 : sub_w;
        const int crop_unit_y = (chroma_array_type == 0 ? 1 : sub_h) * (2 - (s.frame_mbs_only ? 1 : 0));
        s.coded_width = mbs_w * 16;
        s.coded_height = (2 - (s.frame_mbs_only ? 1 : 0)) * map_units_h * 16;
        s.width = s.coded_width - crop_unit_x * (crop_l + crop_r);
        s.height = s.coded_height - crop_unit_y * (crop_t + crop_b);
        if (br.flag()) parse_vui(br, s);
        return s;
    } catch (const ParseError&) {
        return std::nullopt;
    }
}

std::optional<Pps> parse_pps(const uint8_t* nal, size_t size) {
    if (size < 3 || (nal[0] & 31) != kNalPps) return std::nullopt;
    const std::vector<uint8_t> rbsp = unescape_rbsp(nal + 1, size - 1);
    try {
        BitReader br(rbsp.data(), rbsp.size());
        Pps p;
        p.pps_id = static_cast<int>(br.ue());
        p.sps_id = static_cast<int>(br.ue());
        p.cabac = br.flag();
        br.flag();  // bottom_field_pic_order_in_frame_present_flag
        p.num_slice_groups = static_cast<int>(br.ue()) + 1;
        if (p.num_slice_groups > 1) return p;  // FMO is Baseline/Extended only; not parsed further
        p.num_ref_idx_l0_default = static_cast<int>(br.ue()) + 1;
        p.num_ref_idx_l1_default = static_cast<int>(br.ue()) + 1;
        p.weighted_pred = br.flag();
        p.weighted_bipred_idc = static_cast<int>(br.u(2));
        p.pic_init_qp = 26 + br.se();
        br.se();    // pic_init_qs_minus26
        br.se();    // chroma_qp_index_offset
        br.flag();  // deblocking_filter_control_present_flag
        br.flag();  // constrained_intra_pred_flag
        br.flag();  // redundant_pic_cnt_present_flag
        if (br.more_rbsp_data()) p.transform_8x8_mode = br.flag();
        return p;
    } catch (const ParseError&) {
        return std::nullopt;
    }
}

std::optional<SliceHeader> parse_slice_header(const uint8_t* nal, size_t size) {
    const int type = nal[0] & 31;
    if (size < 2 || (type != kNalSlice && type != kNalSliceIdr)) return std::nullopt;
    const std::vector<uint8_t> rbsp = unescape_rbsp(nal + 1, std::min<size_t>(size - 1, 16));
    try {
        BitReader br(rbsp.data(), rbsp.size());
        SliceHeader h;
        h.first_mb = br.ue();
        const uint32_t st = br.ue();
        if (st > 9) return std::nullopt;
        h.type = static_cast<SliceType>(st % 5);  // 5..9 mean "all slices in the picture have this type"
        return h;
    } catch (const ParseError&) {
        return std::nullopt;
    }
}

const char* profile_name(int p) {
    switch (p) {
        case 66: return "Baseline";
        case 77: return "Main";
        case 88: return "Extended";
        case 100: return "High";
        case 110: return "High 10";
        case 122: return "High 4:2:2";
        case 244: return "High 4:4:4 Predictive";
        case 44: return "CAVLC 4:4:4 Intra";
        default: return "Unknown";
    }
}

std::string level_name(int level_idc, int constraint_flags, int profile_idc) {
    // level_idc 11 with constraint_set3 means level 1b for Baseline/Main/Extended.
    if (level_idc == 11 && (constraint_flags & 0x10) && (profile_idc == 66 || profile_idc == 77 || profile_idc == 88))
        return "1b";
    const int major = level_idc / 10, minor = level_idc % 10;
    return minor ? std::to_string(major) + "." + std::to_string(minor) : std::to_string(major);
}

const char* colour_primaries_name(int v) {
    switch (v) {
        case 1: return "BT.709";
        case 5: case 6: return "BT.601";
        case 9: return "BT.2020";
        case 2: return "unspecified";
        default: return v < 0 ? "not signalled" : "other";
    }
}

StreamStats analyze_annexb(const uint8_t* data, size_t size) {
    StreamStats st;
    st.total_bytes = size;
    const std::vector<NalUnit> nals = split_annexb(data, size);

    // A NAL's cost includes its start code; attribute non-VCL NALs (SPS/PPS/SEI)
    // to the frame that follows them.
    size_t pending = 0;
    long last_idr_index = -1;
    for (size_t i = 0; i < nals.size(); ++i) {
        const NalUnit& n = nals[i];
        // Bytes this NAL occupies on the wire: from its start code to the next NAL's start code.
        const size_t bytes = ((i + 1 < nals.size()) ? nals[i + 1].offset : size) - n.offset;

        if (n.type >= 0 && n.type < 32) ++st.nal_counts[n.type];
        if (n.type == kNalSps && !st.sps) st.sps = parse_sps(n.data, n.size);
        if (n.type == kNalPps && !st.pps) st.pps = parse_pps(n.data, n.size);

        if (n.type == kNalSlice || n.type == kNalSliceIdr) {
            const auto sh = parse_slice_header(n.data, n.size);
            if (sh && sh->first_mb == 0) {
                FrameInfo f;
                f.idr = (n.type == kNalSliceIdr);
                f.type = sh->type == SliceType::B ? 'B' : (sh->type == SliceType::P || sh->type == SliceType::SP) ? 'P' : 'I';
                f.bytes = pending + bytes;
                pending = 0;
                st.frames.push_back(f);
                switch (f.type) {
                    case 'I': ++st.i_frames; break;
                    case 'P': ++st.p_frames; break;
                    default: ++st.b_frames; break;
                }
                if (f.idr) {
                    ++st.idr_frames;
                    const long idx = static_cast<long>(st.frames.size()) - 1;
                    if (last_idr_index >= 0) st.gop_lengths.push_back(static_cast<size_t>(idx - last_idr_index));
                    last_idr_index = idx;
                }
            } else if (!st.frames.empty()) {
                st.frames.back().bytes += pending + bytes;  // additional slice of the current picture
                pending = 0;
            } else {
                pending += bytes;
            }
        } else {
            pending += bytes;
        }
    }
    return st;
}

}  // namespace ff::h264
