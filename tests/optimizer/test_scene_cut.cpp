#include "pipeline/scene_cut.hpp"
#include "test_util.hpp"

using namespace ff;

namespace {
FramePtr frame_for(int t, int cut_at) {
    auto f = std::make_shared<Frame>(128, 96);
    if (t < cut_at) {
        fftest::make_moving_frame(*f, t);
    } else {
        fftest::make_noise_frame(*f, 1000 + static_cast<uint32_t>(t / 1000));  // unrelated content
        // keep a little temporal variation inside the new scene
        f->data(0)[t % 50] = uint8_t(t);
    }
    f->pts = t;
    return f;
}
}  // namespace

TEST(finds_a_hard_cut_exactly) {
    std::vector<FramePtr> clip;
    for (int t = 0; t < 30; ++t) clip.push_back(frame_for(t, 15));
    const auto cuts = detect_scene_cuts(clip);
    REQUIRE(cuts.size() == 1);
    CHECK_EQ(cuts[0], 15);
}

TEST(smooth_motion_is_not_a_cut) {
    std::vector<FramePtr> clip;
    for (int t = 0; t < 40; ++t) clip.push_back(frame_for(t, 1000));
    CHECK(detect_scene_cuts(clip).empty());
}

TEST(static_content_is_not_a_cut) {
    std::vector<FramePtr> clip;
    for (int t = 0; t < 20; ++t) {
        auto f = std::make_shared<Frame>(64, 64);
        f->fill(90, 128, 128);
        clip.push_back(f);
    }
    CHECK(detect_scene_cuts(clip).empty());
}

TEST(first_frame_is_never_a_cut) {
    SceneCutDetector d;
    Frame f(64, 64);
    fftest::make_noise_frame(f, 1);
    CHECK(!d.push(f));
}

TEST(two_cuts_are_both_found) {
    std::vector<FramePtr> clip;
    for (int t = 0; t < 12; ++t) clip.push_back(frame_for(t, 6));       // moving scene, then noise at 6
    for (int t = 12; t < 24; ++t) {                                     // jump back to a brand new moving scene at 12
        auto f = std::make_shared<Frame>(128, 96);
        fftest::make_moving_frame(*f, 100 + (t - 12), 777);  // consecutive frames of an unrelated scene
        clip.push_back(f);
    }
    const auto cuts = detect_scene_cuts(clip);
    REQUIRE(cuts.size() == 2);
    CHECK_EQ(cuts[0], 6);
    CHECK_EQ(cuts[1], 12);
}
