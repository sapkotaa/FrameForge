#include "codec/h264_parser.hpp"
#include "test_util.hpp"

using namespace ff::h264;

namespace {

// Writes bits MSB-first, with Exp-Golomb helpers and emulation prevention on output.
class BitWriter {
public:
    void u(uint32_t v, int n) {
        for (int i = n - 1; i >= 0; --i) bit((v >> i) & 1u);
    }
    void flag(bool b) { bit(b ? 1 : 0); }
    void ue(uint32_t v) {
        const uint32_t x = v + 1;
        int len = 0;
        while ((x >> len) > 1) ++len;
        u(0, len);
        u(x, len + 1);
    }
    void se(int32_t v) { ue(v > 0 ? static_cast<uint32_t>(2 * v - 1) : static_cast<uint32_t>(-2 * v)); }
    void trailing() {
        bit(1);
        while (n_ % 8) bit(0);
    }
    // NAL = header byte + escaped RBSP
    std::vector<uint8_t> nal(int ref_idc, int type) const {
        std::vector<uint8_t> out{uint8_t((ref_idc << 5) | type)};
        int zeros = 0;
        for (uint8_t b : bytes_) {
            if (zeros >= 2 && b <= 3) {
                out.push_back(3);
                zeros = 0;
            }
            out.push_back(b);
            zeros = (b == 0) ? zeros + 1 : 0;
        }
        return out;
    }

private:
    void bit(uint32_t b) {
        if (n_ % 8 == 0) bytes_.push_back(0);
        if (b) bytes_.back() |= uint8_t(0x80 >> (n_ % 8));
        ++n_;
    }
    std::vector<uint8_t> bytes_;
    size_t n_ = 0;
};

std::vector<uint8_t> baseline_sps(int mbs_w, int mbs_h) {
    BitWriter w;
    w.u(66, 8);   // profile_idc: Baseline
    w.u(0xC0, 8); // constraint_set0/1
    w.u(30, 8);   // level 3.0
    w.ue(0);      // sps id
    w.ue(0);      // log2_max_frame_num_minus4
    w.ue(2);      // poc type 2
    w.ue(1);      // max_num_ref_frames
    w.flag(false);
    w.ue(uint32_t(mbs_w - 1));
    w.ue(uint32_t(mbs_h - 1));
    w.flag(true);   // frame_mbs_only
    w.flag(true);   // direct_8x8_inference
    w.flag(false);  // no cropping
    w.flag(false);  // no VUI
    w.trailing();
    return w.nal(3, 7);
}

}  // namespace

TEST(bitwriter_expgolomb_known_codes) {
    BitWriter w;
    w.ue(0);  // 1
    w.ue(1);  // 010
    w.ue(2);  // 011
    w.ue(3);  // 00100
    w.trailing();
    const auto n = w.nal(0, 1);
    CHECK_EQ(int(n[1]), 0b10100110);  // 1 010 011 0
    CHECK_EQ(int(n[2] >> 6), 0b01);   // 0100 -> next bits "0100" start: 00100 continues
}

TEST(sps_baseline_640x480) {
    const auto nal = baseline_sps(40, 30);
    const auto s = parse_sps(nal.data(), nal.size());
    REQUIRE(s.has_value());
    CHECK_EQ(s->profile_idc, 66);
    CHECK_EQ(s->level_idc, 30);
    CHECK_EQ(s->width, 640);
    CHECK_EQ(s->height, 480);
    CHECK_EQ(s->max_num_ref_frames, 1);
    CHECK_EQ(s->pic_order_cnt_type, 2);
    CHECK_EQ(s->chroma_format_idc, 1);
    CHECK(std::string(profile_name(s->profile_idc)) == "Baseline");
    CHECK(level_name(s->level_idc, s->constraint_flags, s->profile_idc) == "3");
}

TEST(sps_high_profile_1080p_with_cropping_and_vui) {
    BitWriter w;
    w.u(100, 8);  // High
    w.u(0, 8);
    w.u(40, 8);  // level 4.0
    w.ue(0);
    w.ue(1);      // chroma_format_idc 4:2:0
    w.ue(0);      // bit depth luma 8
    w.ue(0);      // bit depth chroma 8
    w.flag(false);
    w.flag(false);  // no scaling matrix
    w.ue(0);        // log2_max_frame_num_minus4
    w.ue(0);        // poc type 0
    w.ue(2);        // log2_max_poc_lsb_minus4
    w.ue(4);        // 4 reference frames
    w.flag(false);
    w.ue(119);      // 1920 wide
    w.ue(67);       // 1088 tall (68 MBs)
    w.flag(true);
    w.flag(true);
    w.flag(true);   // frame_cropping_flag
    w.ue(0);
    w.ue(0);
    w.ue(0);
    w.ue(4);        // bottom crop: 4 units * CropUnitY(2) = 8 rows
    w.flag(true);   // VUI present
    w.flag(false);  // aspect ratio
    w.flag(false);  // overscan
    w.flag(true);   // video_signal_type_present
    w.u(5, 3);
    w.flag(false);  // limited range
    w.flag(true);   // colour description
    w.u(1, 8);
    w.u(1, 8);
    w.u(1, 8);      // BT.709
    w.flag(false);  // chroma loc
    w.flag(true);   // timing info
    w.u(1001, 32);
    w.u(60000, 32);
    w.flag(true);
    w.trailing();
    const auto nal = w.nal(3, 7);
    const auto s = parse_sps(nal.data(), nal.size());
    REQUIRE(s.has_value());
    CHECK_EQ(s->coded_width, 1920);
    CHECK_EQ(s->coded_height, 1088);
    CHECK_EQ(s->width, 1920);
    CHECK_EQ(s->height, 1080);  // 1088 - 8 cropped rows
    CHECK_EQ(s->max_num_ref_frames, 4);
    CHECK_EQ(s->colour_primaries, 1);
    CHECK_EQ(s->video_full_range, 0);
    CHECK(std::string(colour_primaries_name(s->colour_primaries)) == "BT.709");
    CHECK(s->timing_present);
    CHECK_NEAR(s->fps(), 29.97, 0.01);
    CHECK(std::string(profile_name(100)) == "High");
    CHECK(level_name(40, 0, 100) == "4");
}

TEST(sps_with_scaling_matrix_is_parsed_past) {
    BitWriter w;
    w.u(100, 8);
    w.u(0, 8);
    w.u(31, 8);
    w.ue(0);
    w.ue(1);
    w.ue(0);
    w.ue(0);
    w.flag(false);
    w.flag(true);  // seq_scaling_matrix_present_flag
    for (int i = 0; i < 8; ++i) {
        const bool present = (i == 0 || i == 6);
        w.flag(present);
        if (present) {
            const int size = i < 6 ? 16 : 64;
            // delta_scale: first +2 (next=10), then 0 deltas, then -10 -> next=0 (use default for the rest)
            w.se(2);
            for (int j = 1; j < size - 3; ++j) w.se(0);
            w.se(-10);
        }
    }
    w.ue(0);
    w.ue(2);
    w.ue(1);
    w.flag(false);
    w.ue(19);  // 320
    w.ue(14);  // 240
    w.flag(true);
    w.flag(true);
    w.flag(false);
    w.flag(false);
    w.trailing();
    const auto nal = w.nal(3, 7);
    const auto s = parse_sps(nal.data(), nal.size());
    REQUIRE(s.has_value());
    CHECK_EQ(s->width, 320);
    CHECK_EQ(s->height, 240);
}

TEST(sps_survives_emulation_prevention_bytes) {
    // sps_id and sizes chosen so the RBSP contains 00 00 0x sequences.
    const auto nal = baseline_sps(1, 1);  // 16x16, lots of zero bits
    const auto s = parse_sps(nal.data(), nal.size());
    REQUIRE(s.has_value());
    CHECK_EQ(s->width, 16);
    CHECK_EQ(s->height, 16);
}

TEST(sps_truncated_data_is_rejected_not_crashed) {
    const auto nal = baseline_sps(40, 30);
    for (size_t cut = 0; cut < nal.size(); ++cut) {
        const auto s = parse_sps(nal.data(), cut);  // must never throw or read out of bounds
        if (cut < 9) CHECK(!s.has_value());
    }
    const uint8_t wrong_type[] = {0x68, 0x00, 0x00, 0x00, 0x00, 0x00};
    CHECK(!parse_sps(wrong_type, sizeof wrong_type).has_value());
}

TEST(unescape_removes_emulation_prevention) {
    const uint8_t in[] = {0, 0, 3, 1, 0, 0, 3, 0, 0, 3, 7, 5};
    const auto out = unescape_rbsp(in, sizeof in);
    const std::vector<uint8_t> expect = {0, 0, 1, 0, 0, 0, 0, 7, 5};
    CHECK(out == expect);
}

TEST(split_annexb_handles_3_and_4_byte_start_codes) {
    const uint8_t s[] = {0, 0, 0, 1, 0x67, 0xAA, 0, 0, 1, 0x68, 0xBB, 0, 0, 0, 1, 0x65, 0xCC, 0};
    const auto nals = split_annexb(s, sizeof s);
    REQUIRE(nals.size() == 3);
    CHECK_EQ(nals[0].type, 7);
    CHECK_EQ(nals[0].size, size_t(2));
    CHECK_EQ(nals[0].ref_idc, 3);
    CHECK_EQ(nals[1].type, 8);
    CHECK_EQ(nals[1].size, size_t(2));  // trailing zero_byte of the next 4-byte start code is stripped
    CHECK_EQ(nals[2].type, 5);
    CHECK_EQ(nals[2].size, size_t(2));  // trailing_zero_8bits stripped
    CHECK_EQ(nals[0].offset, size_t(0));
    CHECK_EQ(nals[1].offset, size_t(6));
}

TEST(pps_flags) {
    BitWriter w;
    w.ue(0);
    w.ue(0);
    w.flag(true);   // CABAC
    w.flag(false);
    w.ue(0);        // 1 slice group
    w.ue(3);        // 4 refs
    w.ue(0);
    w.flag(true);   // weighted pred
    w.u(2, 2);
    w.se(-4);       // init qp 22
    w.se(0);
    w.se(0);
    w.flag(true);
    w.flag(false);
    w.flag(false);
    w.flag(true);   // transform_8x8_mode
    w.flag(false);  // pic_scaling_matrix_present
    w.se(0);        // second_chroma_qp_index_offset
    w.trailing();
    const auto nal = w.nal(3, 8);
    const auto p = parse_pps(nal.data(), nal.size());
    REQUIRE(p.has_value());
    CHECK(p->cabac);
    CHECK_EQ(p->num_ref_idx_l0_default, 4);
    CHECK(p->weighted_pred);
    CHECK_EQ(p->weighted_bipred_idc, 2);
    CHECK_EQ(p->pic_init_qp, 22);
    CHECK(p->transform_8x8_mode);
}

TEST(slice_header_types) {
    for (auto [coded, expect, mb] : {std::tuple{7u, SliceType::I, 0u}, {5u, SliceType::P, 0u}, {1u, SliceType::B, 12u}, {0u, SliceType::P, 0u}}) {
        BitWriter w;
        w.ue(mb);
        w.ue(coded);
        w.ue(0);
        w.trailing();
        const auto nal = w.nal(2, coded == 7 ? 5 : 1);
        const auto h = parse_slice_header(nal.data(), nal.size());
        REQUIRE(h.has_value());
        CHECK(h->type == expect);
        CHECK_EQ(h->first_mb, mb);
    }
}

TEST(analyze_counts_frames_and_gops_in_a_synthetic_stream) {
    std::vector<uint8_t> stream;
    auto add = [&](const std::vector<uint8_t>& nal, bool four = true) {
        if (four) stream.push_back(0);
        stream.insert(stream.end(), {0, 0, 1});
        stream.insert(stream.end(), nal.begin(), nal.end());
    };
    auto slice = [](int type, uint32_t slice_type) {
        BitWriter w;
        w.ue(0);
        w.ue(slice_type);
        w.ue(0);
        w.u(0xAB, 8);  // some payload
        w.trailing();
        return w.nal(type == 5 ? 3 : 2, type);
    };
    add(baseline_sps(8, 6));
    BitWriter pw;
    pw.ue(0); pw.ue(0); pw.flag(false); pw.flag(false); pw.ue(0); pw.ue(0); pw.ue(0); pw.flag(false); pw.u(0, 2);
    pw.se(0); pw.se(0); pw.se(0); pw.flag(false); pw.flag(false); pw.flag(false); pw.trailing();
    add(pw.nal(3, 8));
    for (int gop = 0; gop < 3; ++gop) {
        add(slice(5, 7));                        // IDR
        for (int i = 0; i < 4; ++i) add(slice(1, 5), false);  // P frames
    }
    const auto st = analyze_annexb(stream.data(), stream.size());
    CHECK_EQ(st.frames.size(), size_t(15));
    CHECK_EQ(st.idr_frames, size_t(3));
    CHECK_EQ(st.i_frames, size_t(3));
    CHECK_EQ(st.p_frames, size_t(12));
    REQUIRE(st.gop_lengths.size() == 2);
    CHECK_EQ(st.gop_lengths[0], size_t(5));
    CHECK(st.sps.has_value());
    CHECK_EQ(st.sps->width, 128);
    CHECK(st.pps.has_value());
    size_t sum = 0;
    for (const auto& f : st.frames) sum += f.bytes;
    CHECK_EQ(sum, stream.size());  // every byte is attributed to exactly one frame
}
