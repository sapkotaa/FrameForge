# FrameForge

**A media-processing framework in C++20, with a per-title bitrate-ladder optimizer built on top.**

FrameForge takes a video, works out how it behaves at different resolutions and quality levels, chooses the
bitrate "ladder" worth streaming for *that* video, and encodes the rungs in parallel chunks. The pieces underneath
are the building blocks of a transcoding platform: SIMD kernels with runtime CPU dispatch, a lock-free frame
pipeline with backpressure, quality metrics, a retrying job runner, and an H.264 bitstream parser written from the spec.

It builds and runs on **macOS (arm64 / NEON)** and **Linux (x86_64 / AVX2)** from a single codebase.

```
$ frameforge ladder clip.mp4 --out out --encode
```

---

## Contents

1. [Quick start](#1-quick-start)
2. [Installing and building](#2-installing-and-building)
3. [Command-line reference](#3-command-line-reference)
4. [Output files](#4-output-files)
5. [Worked example](#5-worked-example)
6. [Architecture](#6-architecture)
7. [Module reference](#7-module-reference)
8. [Algorithms in detail](#8-algorithms-in-detail)
9. [SIMD and runtime dispatch](#9-simd-and-runtime-dispatch)
10. [Concurrency model](#10-concurrency-model)
11. [Testing and verification](#11-testing-and-verification)
12. [Benchmarks](#12-benchmarks)
13. [Troubleshooting](#13-troubleshooting)
14. [Limitations](#14-limitations)
15. [Roadmap](#15-roadmap)
16. [Glossary](#16-glossary)
17. [Repository layout](#17-repository-layout)

---

## 1. Quick start

```bash
# macOS:   brew install cmake ninja pkg-config ffmpeg
# Ubuntu:  sudo apt install build-essential cmake ninja-build pkg-config ffmpeg \
#              libavformat-dev libavcodec-dev libavutil-dev libswscale-dev

cmake -S . -B build -G Ninja && cmake --build build
ctest --test-dir build --output-on-failure          # expect: 100% tests passed out of 11

tools/make_sample.sh                                 # generates samples/sample_720p.mp4
build/frameforge cpu                                 # which SIMD backend is active
build/frameforge ladder samples/sample_720p.mp4 --frames 0 --out out --encode
```

If the tests pass and the last command ends with `decoded 180/180 frames` on every rung line, everything works.

---

## 2. Installing and building

### Requirements

| Need | Version | Notes |
|---|---|---|
| C++ compiler | C++20. Tested: Apple clang 15 and Homebrew clang 23. GCC 11+ should work but has not been tried yet | `std::bit_ceil`, `<bit>` |
| CMake | ≥ 3.20 | |
| Ninja | any | optional, recommended |
| FFmpeg dev libraries | libavformat, libavcodec, libavutil, libswscale (tested with 8.1) | optional, see below |
| FFmpeg built with `libx264` | | needed by `ladder`; `libx265` needed for `compare` |
| `pkg-config` | | used to find FFmpeg |

The **core library, all kernels, all tests except `test_codec`, and all benchmarks need no FFmpeg at all.**
Configure with `-DFRAMEFORGE_WITH_FFMPEG=OFF` to build just those (the `ladder` and `compare` commands are then
disabled; `cpu` and `analyze` still work).

### macOS

```bash
brew install cmake ninja pkg-config ffmpeg
cmake -S . -B build -G Ninja
cmake --build build
```

Uses NEON automatically on Apple Silicon.

### Linux (Debian / Ubuntu)

```bash
sudo apt install build-essential cmake ninja-build pkg-config ffmpeg \
     libavformat-dev libavcodec-dev libavutil-dev libswscale-dev
cmake -S . -B build -G Ninja
cmake --build build
build/frameforge cpu        # on a modern x86-64 CPU this should report: active backend: avx2
```

Ubuntu's `ffmpeg` package includes libx264 and libx265. On other distros check with
`ffmpeg -encoders | grep -E "libx264|libx265"`.

### CMake options

| Option | Default | Meaning |
|---|---|---|
| `FRAMEFORGE_WITH_FFMPEG` | ON | Build the libav codec layer, probing, chunked encoding and the `ladder`/`compare` commands |
| `FRAMEFORGE_BUILD_APPS` | ON | Build the `frameforge` CLI |
| `FRAMEFORGE_BUILD_TESTS` | ON | Build the unit tests |
| `FRAMEFORGE_BUILD_BENCHMARKS` | ON | Build `bench_luma`, `bench_resize`, `bench_pipeline` |
| `FRAMEFORGE_SANITIZE` | empty | `thread`, or `address,undefined` |
| `CMAKE_BUILD_TYPE` | Release | |

The target architecture decides which SIMD file is compiled: arm64 builds `neon.cpp`, x86_64 builds `avx2.cpp`
(with `-mavx2` on that one file only). On any other architecture only the scalar kernels are built and everything
still works.

### The check script

`tools/check.sh` builds and tests in one step, identically on macOS and Linux:

```bash
tools/check.sh            # release build + tests + prints the active SIMD backend
tools/check.sh scalar     # same tests with SIMD forced off
tools/check.sh tsan       # ThreadSanitizer build
tools/check.sh asan       # AddressSanitizer + UBSan build
tools/check.sh noffmpeg   # core library only
tools/check.sh all        # everything above
```

Each mode builds into its own `build-<mode>/` directory.

---

## 3. Command-line reference

The executable is `build/frameforge`. Running it with no arguments or `--help` prints usage.

### `frameforge ladder <clip> [options]`

Builds a per-title bitrate ladder for a video file (anything libav can open).

| Option | Default | Meaning |
|---|---|---|
| `--frames N` | 90 | Frames to analyze. `0` = the whole clip. All analyzed frames are held in memory. |
| `--crfs a,b,c` | 18,22,26,30,34,38 | CRF values to probe. Lower CRF = higher quality and bitrate. |
| `--heights a,b,c` | standard steps ≤ source | Rendition heights, e.g. `1080,720,540,360,240`. The source height is always included by default. |
| `--codec NAME` | libx264 | `libx264` or `libx265` |
| `--preset NAME` | veryfast | Encoder preset (`ultrafast` … `placebo`) |
| `--jobs N` | all cores | Worker threads for probe and chunk jobs |
| `--max-rungs N` | 6 | Upper bound on ladder size |
| `--out DIR` | none | Write `probes.csv` and `ladder.json` here (and the rung streams with `--encode`) |
| `--encode` | off | After choosing the ladder, encode every rung with chunk-parallel jobs, write `.h264` files, then decode them back and verify. Requires `--out`. |
| `--chunk-min N` | 24 | Minimum chunk length in frames |
| `--chunk-max N` | 48 | Maximum chunk length in frames |
| `--price X` | 0.04 | USD per CPU-hour, used only for the cost line |

What it prints, in order: input info and detected scene cuts → the raw (uncompressed) bitrate → the SIMD backend and
worker count → a table of every probe, tagged `<- rung`, `hull` or `frontier` → the final ladder → probe wall/CPU time,
parallel efficiency, retries → analysis cost per video minute → (with `--encode`) one verification line per rung.

### `frameforge compare <clip> [options]`

Compares two encoder configurations with **BD-rate** (the average bitrate difference at equal quality).

| Option | Default |
|---|---|
| `--a-codec` / `--a-preset` | libx264 / medium (the anchor) |
| `--b-codec` / `--b-preset` | libx265 / medium (the test) |
| `--frames N` | 60 |
| `--crfs a,b,c,d` | 20,25,30,35 |
| `--jobs N` | all cores |

Result is a percentage: **negative means B needs fewer bits than A for the same PSNR**.

### `frameforge analyze <file.h264>`

Inspects a raw Annex B H.264 stream: profile and level, coded and cropped size, chroma format and bit depth,
reference-frame count, CABAC/CAVLC, colour primaries and range from the VUI, frame rate from the VUI, NAL unit
counts, I/P/B/IDR frame counts, GOP length statistics, average I/P/B frame sizes, bitrate, and the first 72 frames'
type pattern in decode order (`R` marks an IDR).

To analyze an MP4:

```bash
ffmpeg -i in.mp4 -an -c:v copy -bsf:v h264_mp4toannexb out.h264
frameforge analyze out.h264
```

### `frameforge cpu`

Prints the architecture, hardware thread count, the SIMD backends this build and CPU support, and which one is active.

```
arch:            arm64
hardware threads: 8
simd backends:   scalar neon
active backend:  neon  (override with FRAMEFORGE_ISA=scalar|neon|avx2)
libav codec layer: enabled
```

### Environment variable

`FRAMEFORGE_ISA=scalar|neon|avx2` forces a SIMD backend (ignored if that backend is unavailable). Useful for
benchmarking against the scalar baseline and for debugging a suspected SIMD bug:
`FRAMEFORGE_ISA=scalar build/frameforge ladder …`.

### Other binaries

| Binary | Purpose |
|---|---|
| `build/test_*` | One test executable per suite. Pass a substring to run matching cases only: `build/test_kernels blend`. |
| `build/bench_luma` | SAD / SSD / plane-sum kernels per SIMD backend |
| `build/bench_resize` | Downscale, blend and full resize per backend |
| `build/bench_pipeline` | Ring buffer vs mutex queue, frame pool, end-to-end pipeline throughput |
| `tools/make_sample.sh [out.mp4]` | Generates a 6 s 720p30 three-scene test clip with ffmpeg |

---

## 4. Output files

With `--out DIR`:

### `ladder.json`

```json
{
  "source": "samples/sample_720p.mp4",
  "width": 1280, "height": 720, "fps": 30, "frames": 180,
  "codec": "libx264", "preset": "veryfast",
  "rungs": [
    {"width": 428,  "height": 240, "crf": 38, "kbps": 53,   "psnr": 33.39,   "ssim": 0.925906},
    {"width": 428,  "height": 240, "crf": 26, "kbps": 118,  "psnr": 37.3542, "ssim": 0.946814},
    {"width": 1280, "height": 720, "crf": 18, "kbps": 4417, "psnr": 47.811,  "ssim": 0.987454}
  ]
}
```

`kbps`, `psnr` and `ssim` are the *probe* measurements for that rung (average bitrate over the clip; mean per-frame
luma PSNR/SSIM versus the source after upscaling). Each rung is a ready-made encoder setting: resolution plus CRF.

### `probes.csv`

One row per probe encode, suitable for plotting the rate–quality curves:

```
width,height,crf,kbps,psnr,ssim,cpu_seconds,on_frontier,on_hull,on_ladder
428,240,38,52.8733,33.39,0.925906,0.378607,1,1,1
428,240,34,58.2413,34.7035,0.932189,0.583915,1,1,0
```

`on_frontier`, `on_hull`, `on_ladder` are 0/1 flags (see [section 8.4](#84-from-probes-to-a-ladder)).

### `rung_<height>p_crf<crf>.h264` (with `--encode`)

Raw Annex B H.264 elementary streams, one per rung, produced by the chunk-parallel encoder. Play with
`ffplay out/rung_720p_crf18.h264`, inspect with `frameforge analyze`, or `ffprobe`.

---

## 5. Worked example

`tools/make_sample.sh` makes a 6-second 1280×720 clip with three scenes: a mandelbrot zoom (fine detail), smooth
gradients (cheap), and a test pattern with light grain (moderately hard).

```
$ build/frameforge ladder samples/sample_720p.mp4 --frames 0 --out out --encode

FrameForge ladder
  input    samples/sample_720p.mp4: 1280x720 @ 30.00 fps, 180 frames (6.0 s), 2 scene cut(s)
  cuts at  frame 60 (2.00 s)  frame 120 (4.00 s)
  raw      332 Mbps uncompressed 8-bit 4:2:0
  runtime  simd=neon, 8 worker threads
  probing  4 renditions x 6 CRFs = 24 encodes (libx264, preset veryfast)

  bitrate ladder (6 rungs; 24 probes -> 13 on the Pareto frontier -> 5 on the convex hull)
    #1   428x240   crf 38       53 kbps   PSNR 33.39 dB   SSIM 0.9259
    #2   428x240   crf 26      118 kbps   PSNR 37.35 dB   SSIM 0.9468
    #3   428x240   crf 22      212 kbps   PSNR 38.47 dB   SSIM 0.9511
    #4   640x360   crf 22      520 kbps   PSNR 39.56 dB   SSIM 0.9615
    #5  1280x720   crf 26     1601 kbps   PSNR 42.63 dB   SSIM 0.9795
    #6  1280x720   crf 18     4417 kbps   PSNR 47.81 dB   SSIM 0.9875

  probe wall 5.9 s, cpu 42.8 s (7.3x parallel), 27 jobs, 0 retries
  analysis cost: $0.0048 per video minute at $0.04/CPU-hour

  chunked encode: 5 chunks (24..48 frames, split at scene cuts when possible)
     240p    63 kbps  wall 0.04s  cpu 0.14s  (3.2x)  decoded 180/180 frames ...
     ...
     720p  4456 kbps  wall 0.65s  cpu 1.94s  (3.0x)  decoded 180/180 frames, PSNR vs source rendition 47.83 dB
```

How to read it:

* **Scene cuts at frames 60 and 120** match where the three scenes were joined.
* **Resolution rises with bitrate.** Below ~300 kbps a sharp 240p beats a starved 720p; above ~1.5 Mbps 720p wins.
  A fixed ladder cannot know this; per-title probing does.
* **`24 probes → 13 → 5`**: of 24 encodes, 13 are non-dominated (the Pareto frontier) and 5 sit on the convex hull.
* **7.3× parallel** means the 27 jobs kept ~7 of 8 cores busy.
* **Chunked bitrates differ slightly from probes** (63 vs 53 kbps at 240p, 4456 vs 4417 at 720p): each chunk starts on
  an IDR frame and repeats SPS/PPS, which costs relatively more at very low bitrates.
* **`PSNR vs source rendition`** on the verify lines compares each rung to its own resolution's source. The probe
  PSNR compares against the full-resolution source after upscaling, so the two are not directly comparable.

---

## 6. Architecture

### Data flow

```
                         ┌─────────────────────────── load_clip ───────────────────────────┐
  clip.mp4 ──► Decoder ──┤ source ─► scene-cut detector ─► collect (frames kept in memory) │
  (libav, I420)          └──────────── FramePipeline: bounded SPSC queues, FramePool ──────┘
                                                       │ frames + scene cut indices
                                                       ▼
                                  JobRunner: scale jobs  (one per rendition height)
                                                       ▼
                                  JobRunner: probe jobs  (one per height × CRF)
                             encode ─► decode ─► upscale to source size ─► PSNR, SSIM
                                                       ▼
                         RatePoint[ ] ─► Pareto frontier ─► convex hull ─► knee ─► rungs
                                                       ▼
                                  plan_chunks (split at scene cuts, min/max length)
                                                       ▼
                       JobRunner: one encode job per chunk ─► concatenate Annex B ─► verify
```

### Layers

```
apps/frameforge.cpp          CLI
optimizer/                   ladder, probe, chunked encode, job runner      (probe + chunked encode need libav)
quality/                     PSNR, SSIM, BD-rate
codec/                       H.264 parser (pure C++) │ Encoder/Decoder (libav, behind PIMPL)
pipeline/                    Frame, ObjectPool, SPSC ring buffer, Channel, FramePipeline, scene cut
kernels/                     scalar │ NEON │ AVX2 behind a runtime dispatch table; resize; yuv
```

Two static libraries: **`frameforge_core`** (everything that does not need libav) and **`frameforge_codec`**
(libav wrappers, probing, chunked encoding). libav types never appear in public headers, so the rest of the code
compiles and tests without FFmpeg installed.

---

## 7. Module reference

### `kernels/` – SIMD kernels and resizing

| Header | Provides |
|---|---|
| `dispatch.hpp` | `Isa`, `KernelTable` of function pointers, `kernels()`, `kernels_for(isa)`, `supported_isas()`, `set_active_isa()` |
| `luma.hpp` | Wrappers: `sad`, `ssd`, `luma_sum`, `luma_mean` over the active backend |
| `resize.hpp` | `downscale2x`, `resize_bilinear`, `resize_plane`, `resize_frame` |
| `yuv.hpp` | `i420_frame_bytes`, `raw_bitrate_mbps`, `rgb24_to_i420` (BT.709, limited range) |

Kernel table entries (all 8-bit samples, strides in bytes):

| Kernel | Definition |
|---|---|
| `sad(a,b,w,h)` | Σ \|a − b\| over a w×h block |
| `ssd(a,b,w,h)` | Σ (a − b)² |
| `sum(p,w,h)` | Σ p |
| `blend_rows(a,b,dst,n,wb)` | `dst = (a·(256−wb) + b·wb + 128) >> 8`, `wb ∈ [0,255]` |
| `downscale2x_row(r0,r1,dst,n)` | `dst = (r0[2i]+r0[2i+1]+r1[2i]+r1[2i+1]+2) >> 2` |

### `pipeline/` – frames and threading

| Header | Provides |
|---|---|
| `frame.hpp` | `Frame` (I420, 64-byte-aligned planes, even dimensions), `FramePtr`, `FramePool`, `make_frame_pool` |
| `buffer_pool.hpp` | `ObjectPool<T>`: preallocated objects, blocking `acquire()`, `try_acquire()`, `close()`, stall counter; objects return to the pool when the last `shared_ptr` drops |
| `ring_buffer.hpp` | `SpscRingBuffer<T>`, `Channel<T>` (blocking, close, cancel, drain, stall counters), `Backoff`, `cpu_relax` |
| `pipeline.hpp` | `FramePipeline`: source → named stages → sink, one thread per stage, per-stage `StageStats` |
| `scene_cut.hpp` | `SceneCutDetector`, `detect_scene_cuts()` |

### `codec/`

| Header | Provides |
|---|---|
| `h264_parser.hpp` | `split_annexb`, `unescape_rbsp`, `parse_sps/pps/slice_header`, `analyze_annexb`, `profile_name`, `level_name` |
| `encoder.hpp` | `Encoder` (libx264/libx265), `EncodeParams`, `PacketView`, `encode_roundtrip()`, `set_codec_log_quiet()` |
| `decoder.hpp` | `Decoder` (file → I420 frames), `PacketDecoder` (bare packets or raw Annex B streams) |

### `quality/`

| Header | Provides |
|---|---|
| `psnr.hpp` | `psnr_y`, `psnr_yuv`, `psnr_plane`, `psnr_from_mse` (capped at 100 dB) |
| `ssim.hpp` | `ssim_y`, `ssim_plane` |
| `bdrate.hpp` | `bd_rate(anchor, test)` |

### `optimizer/`

| Header | Provides |
|---|---|
| `ladder.hpp` | `RatePoint`, `pareto_frontier`, `convex_hull`, `LadderConfig`, `select_ladder`, `cost_per_video_minute`, `break_even_views` |
| `job_runner.hpp` | `JobRunner`: thread pool with retry, futures, `wait_idle`, `Stats` |
| `chunked.hpp` | `Chunk`, `plan_chunks`, `encode_chunked`, `ChunkedResult` |
| `probe.hpp` | `SourceClip`, `load_clip`, `default_ladder_heights`, `ProbeConfig`, `probe_rate_quality` |

---

## 8. Algorithms in detail

### 8.1 Rate–quality measurement

For each (height, CRF) pair a probe job:

1. encodes the (scaled) source frames with `EncodeParams{codec, preset, crf}`,
2. feeds every packet straight into a decoder (no disk, no container),
3. **upscales each reconstructed frame back to the source size** with FrameForge's own resizer,
4. computes luma PSNR and SSIM against the original source frame,
5. reports `kbps = bytes × 8 / 1000 / (frames / fps)` and the mean PSNR/SSIM over frames.

Measuring against the full-resolution source is what lets renditions of different sizes share one quality axis. The
alternative, comparing a 240p encode to a 240p copy of the source, would make every rendition look excellent.

Scaled copies of the source are made once per height (one job each) and shared by that height's probes.

### 8.2 Quality metrics

* **PSNR** = 10·log₁₀(255² / MSE), MSE from the SIMD SSD kernel. Identical frames are reported as 100 dB.
  The clip figure is the **mean of per-frame PSNR**, matching ffmpeg's per-frame `psnr_y` log.
* **SSIM** uses 8×8 windows sampled every 4 pixels, uniform weights, C1 = (0.01·255)², C2 = (0.03·255)². The plane is
  first reduced to 4×4 block sums (Σa, Σb, Σa², Σb², Σab); each window is the sum of four blocks, so every pixel is read
  once. Planes smaller than 8×8 are treated as one window.
* **BD-rate**: fit log₁₀(rate) as a cubic polynomial of quality for each curve (least squares via normal equations;
  lower degree if fewer than four points), integrate both over their overlapping quality range, and convert the mean
  gap to a percentage. Returns NaN if the quality ranges do not overlap.

### 8.3 Scene-cut detection

1. Take the luma plane and box-filter it down 4× (two passes of the SIMD `downscale2x`).
2. Compute the mean absolute difference to the previous frame with the SIMD SAD kernel.
3. Declare a cut if `score > 20` **and** `score > 3 × max(recent average, 1)`.

The absolute threshold ignores noise; the relative one means fast but steady motion (high, constant difference) is not
a cut. History is cleared at a cut so the new scene sets its own baseline. The first frame is never a cut.

### 8.4 From probes to a ladder

Let a probe be a point (log₂ kbps, PSNR).

1. **Pareto frontier**: sort by bitrate; keep a point only if its quality beats every cheaper point. Anything else is
   dominated: you could spend fewer bits for the same or better quality.
2. **Convex hull**: the upper concave envelope of the frontier (monotone-chain algorithm). Slopes strictly decrease:
   diminishing returns.
3. **Knee**: walk the hull and stop at the first segment whose slope falls below `min_db_per_doubling` (default 0.75 dB
   per doubling of bitrate). Nothing above the knee is offered (capped further by `max_kbps`).
4. **Greedy rungs** from the frontier, cheapest first: keep a point only if it adds at least `min_gap_db` (1.0 dB) of
   quality **and** at least `min_bitrate_ratio` (1.25×) of bitrate over the previous rung.
5. **Thinning**: if more than `max_rungs` (6) survive, keep the cheapest and the knee and spread the rest evenly in
   log-bitrate. Thinning only widens gaps, so the gap/ratio guarantees still hold.

Why the frontier, not just the hull, supplies the rungs: with a coarse CRF grid the hull discards real, non-dominated
encodes that sit slightly under a long hull segment. An early version selected from the hull alone and produced a
ladder that jumped from 118 kbps to 4.4 Mbps. The hull is used where it is informative: finding the knee.

All thresholds live in `LadderConfig`.

### 8.5 Chunked parallel encoding

`plan_chunks(n_frames, scene_cuts, min_len, max_len)`:

* walk forward from the chunk start; a chunk ends at the first scene cut that leaves it at least `min_len` frames,
  or at `max_len` frames, whichever comes first;
* chunks are contiguous, cover every frame exactly once, are never longer than `max_len`, and are at least `min_len`
  long except possibly the last.

`encode_chunked` then runs one encoder per chunk on the `JobRunner`:

* `keyint = chunk length`, so a chunk is one GOP that starts on an IDR; there is no open GOP, so no frame references
  across a chunk boundary;
* libx264 repeats SPS/PPS at every keyframe, so each chunk stands alone and the concatenation is a valid stream;
* the same constant CRF in every chunk keeps quality consistent;
* each job rebuilds its output from scratch into its own slot (**idempotent**), so a retry after a failed worker is safe.

The cost of independence: extra IDR frames and repeated headers. Measured on the sample: ~19% more bits at a 53 kbps
rung, ~1% at 4.4 Mbps.

### 8.6 Resizing

`resize_plane` first halves with the 2×2 box filter while the target is at most half the current size (cheap
anti-aliasing for big downscales), then does one fixed-point bilinear step. The bilinear step runs the **vertical
blend first** (full source width, SIMD) and the **horizontal gather second** (scalar, one row per output row);
the opposite order spent ~80% of the time in the scalar pass. Sampling is center-aligned with 8-bit weights.

### 8.7 H.264 parsing

`split_annexb` handles 3- and 4-byte start codes and strips trailing zero bytes. `unescape_rbsp` removes emulation-prevention
bytes (`00 00 03` → `00 00`). The SPS parser covers profile/level, chroma format, bit depth, scaling lists (skipped
correctly), POC type, reference frames, frame size with **cropping** (1088 → 1080), and VUI aspect ratio, colour
description and timing. `analyze_annexb` assigns every byte of the stream to a frame (non-VCL NALs go to the frame that
follows) and reconstructs frame types from slice headers, so I/P/B/IDR sizes and GOP lengths are exact.

---

## 9. SIMD and runtime dispatch

Every kernel has three implementations behind one function-pointer table:

| Backend | File | Compiled when | Selected when |
|---|---|---|---|
| scalar | `kernels/scalar.cpp` | always (auto-vectorization disabled) | always available; the correctness oracle and benchmark baseline |
| NEON | `kernels/neon.cpp` | target is arm64 | always (NEON is mandatory on AArch64) |
| AVX2 | `kernels/avx2.cpp` | target is x86_64 (file built with `-mavx2`) | `__builtin_cpu_supports("avx2")` is true at startup |

Key implementation points:

* **Bit-exact by construction and by test.** Rounding is part of each kernel's contract; the tests require every backend
  to match the scalar one on random data over awkward widths (0, 1, 7, 8, 15, 16, 17, 33, 100, …), strided blocks,
  and extreme values (0 vs 255, the case that overflows an under-sized accumulator).
* **SAD**: NEON `vabdq_u8` + pairwise widening accumulate; AVX2 `vpsadbw`.
* **SSD**: NEON widening multiply (`vmull`), AVX2 `vpmaddwd`; accumulators widened to 64-bit per row so no overflow.
* **Blend**: NEON `vmull/vmlal` then `vrshrn` (= `(x+128)>>8`); AVX2 16-bit lanes (`a·w0 + b·w1 ≤ 65 280`, never overflows
  because weights sum to 256). `wb = 0` is a plain copy because 256 does not fit in a byte.
* **Downscale**: NEON `vpaddlq` + `vpadalq` + `vrshrn`; AVX2 `vpmaddubsw` with ones, plus a lane-order fix after `packus`.
* **Tails** (widths not divisible by the vector size) are finished in scalar inside every kernel.
* `avx2.cpp` is the only file built with `-mavx2`; nothing else in the library requires AVX2, so the binary still runs
  on older x86 CPUs and simply selects scalar.

> **Status of the AVX2 backend:** it compiles (verified with an x86_64 cross-build) and the instructions are present in the
> object code, but it was developed on an Apple M1 and has **not been executed on real AVX2 hardware yet**. On a Linux
> x86 machine run `tools/check.sh`; any mistake appears as a bit-exactness failure naming the kernel.

---

## 10. Concurrency model

### SPSC ring buffer (`SpscRingBuffer<T>`)

* Two atomics only: `head_` (consumer-written) and `tail_` (producer-written), monotonically increasing; capacity is a
  power of two so indexing is a mask.
* The producer writes the slot, then publishes with a **release** store to `tail_`. The consumer's **acquire** load of
  `tail_` that sees the new value also sees the slot contents. The mirrored pairing on `head_` lets the producer reuse a slot.
* Producer-owned and consumer-owned state sit on separate **64-byte cache lines** (no false sharing), and each side
  keeps a cached copy of the other's index, re-reading the shared line only when the cache says full/empty.
* No compare-and-swap, so no ABA problem. `try_push` moves from its argument only on success.
* **Exactly one producer thread and one consumer thread per queue.**

### `Channel<T>`

A blocking wrapper: `push` waits when full (this is the backpressure), `pop` waits when empty. `close()` signals end of
stream (consumers drain what is left, then get `nullopt`); `cancel()` aborts both sides. Waiting uses spin → yield →
50 µs sleep. Stall counters record how often each side had to wait.

### `ObjectPool<T>` / `FramePool`

A fixed set of preallocated frames. `acquire()` blocks when none are free, so a slow consumer bounds memory instead of
letting queues grow. Frames hand themselves back through the `shared_ptr` deleter, so ownership just moves between stages with no
copies. Reuse is LIFO (most recently used frame is warmest in cache). A frame keeps its pool alive until it is released.

### `FramePipeline`

`source → stage₁ → … → stageₙ → sink`, one thread per stage, bounded channels between them. The sink runs on the caller's
thread. Per-stage `StageStats`: frames in/out, busy time, input and output stalls. The stage with ~100% busy time and
neighbours that are starved is the bottleneck.

**Error handling.** If any callback throws, every channel is cancelled, each consumer drains its queue, and the first
exception is rethrown from `run()` after all threads join. A source that can block somewhere other than a queue
(typically `pool->acquire()`) must be woken too, or the join would wait forever, so `run()` takes an `on_cancel` hook:

```cpp
pipeline.run(source, sink, [&] { pool->close(); });
```

### `JobRunner`

A thread pool with a queue. `submit(name, fn, max_attempts=3)` returns a `std::future<void>`; a job that throws is
retried up to `max_attempts`, then the exception is delivered through the future. Stats: submitted, completed, failed,
retries, summed busy time, max queue depth. **Jobs must be idempotent**: write results by overwriting a fixed slot,
never by appending.

---

## 11. Testing and verification

### Test suites (98 cases, ~280 assertions, 11 executables)

| Suite | What it covers |
|---|---|
| `test_kernels` | Scalar vs naive loops; every backend vs scalar; extreme values; `blend_rows` for all 256 weights; `downscale2x_row` |
| `test_resize` | Identity, constant images, monotonic gradients, identical output on every backend, frame-level resize, luma mean, raw bitrate (1080p30 = 746.5 Mbps), RGB→I420 corners |
| `test_ring` | Capacity rounding, FIFO, wraparound, failed push keeps the value, 3 M-item two-thread stress with payload check, channel close/cancel/backpressure |
| `test_pool` | Alignment, odd-size rejection, recycling, blocking and wake-up, close, frames outliving the pool handle, contention |
| `test_pipeline` | Order preserved, dropping frames, backpressure bound (never more frames in flight than the pool holds), exceptions from a stage and from the sink, per-stage timing |
| `test_quality` | PSNR formula and 100 dB cap, identical across backends, SSIM behaviour, BD-rate recovering a known −20% / +25% / −50% |
| `test_ladder` | Frontier and hull properties (random sets), knee, frontier points below the hull, bounds, thinning, strictly increasing ladders, cost model |
| `test_jobs` | Parallel execution, retry of flaky jobs, permanent failure surfaced, idempotent retry, speedup, drain on destruction; chunk-plan invariants over 500 random inputs |
| `test_scenecut` | Hard cut found exactly, smooth motion and static content not flagged, two cuts, first frame |
| `test_h264` | Hand-built SPS/PPS/slice bitstreams (baseline, High 1080p with cropping + VUI, scaling matrices, emulation prevention, truncation at every length), start-code splitting, byte accounting |
| `test_codec` | (needs FFmpeg) Round trips, CRF monotonicity, parser against real x264 output, raw stream decoding, file decoding, chunked-stream validity and determinism, probe grid sanity |

Run one suite or case: `build/test_h264`, `build/test_kernels blend`.

### Beyond unit tests

* **ThreadSanitizer** (`tools/check.sh tsan`) over every suite. As a negative control, weakening the ring buffer's
  release/acquire to relaxed makes TSan report a data race; the shipped ordering is clean.
* **AddressSanitizer + UBSan** (`tools/check.sh asan`).
* **Forced-scalar run** (`tools/check.sh scalar`) so the reference path is tested in its own right.
* **Independent ground truth:** the PSNR FrameForge reports for the stitched 720p rung (47.83 dB) equals ffmpeg's own
  `psnr` filter on the same two files; `ffprobe` decodes the stitched stream to the expected 180 frames; `analyze`
  agrees on profile, size and GOP structure.

---

## 12. Benchmarks

Apple M1 (8 cores), Release build, 1080p planes. Run `build/bench_*` on your machine; on x86 the AVX2 column appears automatically.

### Kernels (GB/s of input processed)

| Kernel | scalar | NEON | speedup |
|---|---:|---:|---:|
| SAD 1920×1080 | 4.39 | 48.31 | **11.0×** |
| SSD 1920×1080 | 3.60 | 21.22 | 5.9× |
| Plane sum | 3.05 | 26.22 | 8.6× |
| SAD 16×16 blocks (motion-search shape) | 4.43 | 24.51 | 5.5× |

Small blocks gain less (an 8×16 block is ~6.7×, a 128×16 block ~14×): call overhead and tails matter.

### Resize (time per call)

| Case | scalar | NEON | speedup |
|---|---:|---:|---:|
| `blend_rows`, 1920-px row | 1.2 µs | 0.1 µs | 15.6× |
| `downscale2x` 1080p → 540p | 315.5 µs | 31.6 µs | 10.0× |
| `resize_plane` 1080p → 720p | 1429.9 µs | 673.8 µs | 2.1× |
| `resize_plane` 1080p → 360p | 675.6 µs | 203.2 µs | 3.3× |
| `resize_frame` I420 1080p → 720p | 2354.6 µs | 1031.4 µs | 2.3× |

The end-to-end resize gains less than its SIMD kernels because the horizontal gather is scalar (Amdahl's law).
Reordering to vertical-first cut 1080p → 720p on NEON from 1.95 ms to 0.67 ms.

### Pipeline and queues

| | |
|---|---|
| 1080p luma-mean → downscale pipeline, scalar kernels | ~1 400 fps |
| same, NEON kernels | ~11 700 fps |
| SPSC queue, capacity 4: lock-free vs mutex+condvar | 22.6 vs 1.0 M items/s |
| SPSC queue, capacity 1024 | 35.7 vs 16.0 M items/s |
| Frame from pool vs fresh 1080p allocation | 49 ns vs 141 µs |

### Ladder run (sample clip, 24 probes + 6 chunked rung encodes)

~6 s probe wall time, 7.3× parallel efficiency on 8 cores; chunked rung encodes 3.0–3.3× faster than their summed CPU time.

---

## 13. Troubleshooting

**`encoder not available in this libav build: libx264`** — your FFmpeg lacks libx264. Check
`ffmpeg -encoders | grep libx264`; install a full build (`brew install ffmpeg`, Ubuntu's `ffmpeg` package). `--codec libx265`
needs libx265 likewise.

**CMake cannot find libavformat** — install the dev packages and `pkg-config` (see section 2); or build without
them using `-DFRAMEFORGE_WITH_FFMPEG=OFF`.

**`this build has no libav support`** — you configured with FFmpeg off; reconfigure with `-DFRAMEFORGE_WITH_FFMPEG=ON`.

**`active backend: scalar` on a CPU that should have SIMD** — check `FRAMEFORGE_ISA` is not set in your environment;
on x86 verify the CPU reports AVX2 (`grep avx2 /proc/cpuinfo`). Running x86_64 binaries under Rosetta 2 on older
macOS has no AVX2.

**ASan/UBSan tests abort with "terminating due to uncaught exception"** — seen with Homebrew LLVM clang 23: UBSan's
`function` check breaks C++ exception unwinding. The build already adds `-fno-sanitize=function` for Clang; if you
override the flags yourself, keep it.

**Sanitizer or timing-sensitive tests time out on a busy machine** — the concurrency tests include sleeps and
thousands of thread hand-offs; a heavily loaded system can exceed `ctest` timeouts. Re-run when idle.

**`ladder` is slow or uses a lot of memory** — all analyzed frames live in RAM (a 1080p frame is ~3 MB, plus scaled
copies). Lower `--frames`, `--heights` and `--crfs`, or use a lower-resolution clip.

**Odd-sized video** — dimensions are rounded down to even (I420 requires it); rendition widths are rounded to even.

**`analyze` says "no SPS found"** — the file is not Annex B (e.g. an MP4). Extract the stream with the `ffmpeg … h264_mp4toannexb`
command shown in section 3.

---

## 14. Limitations

* **AVX2 not yet run on real hardware** (see section 9). NEON, scalar and everything else are tested.
* **PSNR and SSIM, not VMAF.** Production ladders are usually tuned with VMAF. The ffmpeg used in development has
  `libvmaf`, so cross-checking the chosen rungs with `ffmpeg -lavfi libvmaf` is straightforward.
* **Synthetic sample clip.** The included clip exercises the code; conclusions about codecs or content types
  (e.g. the `compare` result of +0.4% for x265 vs x264) should come from real footage.
* **In-memory analysis.** Targets short clips and per-shot analysis, not hour-long VOD.
* **Codecs and formats:** H.264 and HEVC via libx264/libx265; Annex B elementary streams out; no MP4/HLS/DASH packaging;
  8-bit 4:2:0 only; no audio; the stream parser covers H.264 only.
* **Whole-clip ladder.** One ladder per clip (per-title), not per shot.
* The horizontal pass of the resizer is scalar.
* Frame-rate handling assumes constant frame rate.

---

## 15. Roadmap

1. Run and benchmark the AVX2 backend on Linux; add an SSE4.1 fallback.
2. VMAF as the quality metric (via libvmaf) alongside PSNR/SSIM.
3. Vectorize the horizontal resize pass (NEON `vtbl`, AVX2 gathers or shuffles).
4. Per-shot ladders: probe each scene separately and choose parameters per shot.
5. Hardware encoders (VideoToolbox on macOS, NVENC/VAAPI on Linux) as probe targets, with a software-vs-hardware cost table.
6. SVT-AV1 / AV1 support and an OBU parser next to the NAL parser.
7. HLS/DASH packaging (fMP4 segments + manifests) from the encoded rungs.
8. Streaming input (decode on demand) so memory stays bounded for long videos.
9. A worker-pool abstraction over processes or machines using the same idempotent job contract.
10. ONNX-based detection for ROI-weighted encoding and content classification.

---

## 16. Glossary

| Term | Meaning |
|---|---|
| **I420 / 4:2:0** | Planar YCbCr with chroma at half resolution both ways: 1.5 samples per pixel |
| **CRF** | Constant Rate Factor: quality-targeted encoding; bitrate varies with content. Lower = better |
| **GOP / IDR** | Group of pictures / instantaneous decoder refresh: a keyframe that resets references |
| **Annex B** | H.264 byte-stream format with `00 00 01` start codes (vs length-prefixed AVCC in MP4) |
| **NAL unit** | The packet unit of H.264; types include SPS, PPS, SEI, slices |
| **SPS / PPS** | Sequence / picture parameter sets: resolution, profile, coding tools |
| **Emulation prevention** | `03` bytes inserted so payload bytes never look like a start code |
| **Bitrate ladder** | The set of (resolution, bitrate) renditions offered to a player |
| **Per-title encoding** | Choosing the ladder from measurements on the specific video |
| **Pareto frontier** | Points not dominated by any other point (no cheaper-and-better alternative) |
| **Convex hull** | Upper concave envelope of the rate–quality points |
| **Knee** | Where extra bits stop buying meaningful quality |
| **BD-rate** | Bjøntegaard delta-rate: average bitrate difference between two codecs/settings at equal quality |
| **PSNR / SSIM** | Pixel-error and structural-similarity quality metrics |
| **SAD / SSD** | Sum of absolute / squared differences |
| **SIMD** | One instruction operating on many data lanes (NEON 128-bit, AVX2 256-bit) |
| **Backpressure** | A slow stage throttling upstream instead of letting queues grow |
| **SPSC** | Single-producer single-consumer |
| **False sharing** | Two cores fighting over one cache line holding unrelated variables |
| **Idempotent job** | A job that can be re-run safely and produces the same result |

---

## 17. Repository layout

```
CMakeLists.txt
README.md
apps/frameforge.cpp                 CLI: ladder | compare | analyze | cpu
include/
  kernels/    dispatch.hpp luma.hpp resize.hpp yuv.hpp
  pipeline/   frame.hpp buffer_pool.hpp ring_buffer.hpp pipeline.hpp scene_cut.hpp
  codec/      h264_parser.hpp encoder.hpp decoder.hpp
  quality/    psnr.hpp ssim.hpp bdrate.hpp
  optimizer/  ladder.hpp job_runner.hpp chunked.hpp probe.hpp
src/
  kernels/    scalar.cpp neon.cpp avx2.cpp dispatch.cpp impl.hpp resize.cpp yuv.cpp
  pipeline/   frame.cpp pipeline.cpp scene_cut.cpp
  codec/      h264_parser.cpp encoder.cpp decoder.cpp
  quality/    psnr.cpp ssim.cpp bdrate.cpp
  optimizer/  ladder.cpp job_runner.cpp chunk_plan.cpp probe.cpp chunked_encode.cpp
tests/        test_util.hpp test_main.cpp + kernels/ pipeline/ quality/ optimizer/ codec/
benchmarks/   bench_util.hpp luma.cpp resize.cpp pipeline.cpp
tools/        make_sample.sh check.sh
samples/      generated test clip (git-ignored)
out/          generated output (git-ignored)
```
