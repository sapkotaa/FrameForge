# FrameForge

A small media-processing framework in C++20 with a **per-title bitrate-ladder optimizer** on top.

```
$ frameforge ladder clip.mp4 --out out --encode
```

It decodes a clip, probes how it behaves at several resolutions and quality levels, finds the rungs worth
shipping for *that* video, and encodes them with chunk-parallel jobs. Underneath is the infrastructure a
transcoding platform needs: SIMD kernels with runtime dispatch, a lock-free frame pipeline with backpressure,
quality metrics, a job runner with retries, and a from-scratch H.264 bitstream parser.

Runs on **macOS (arm64, NEON)** and **Linux (x86_64, AVX2)** from one codebase.

```
              ┌────────────────────────── frameforge ladder ──────────────────────────┐
 clip.mp4 ──► │ decode ─► scene-cut ─► collect │   probe (res × CRF, parallel jobs)    │
              │ (libav)    (SIMD)      (pool)  │   encode ► decode ► upscale ► PSNR/SSIM│
              └─────────────────────────────── └──────────────┬─────────────────────────┘
                                                              ▼
                              Pareto frontier ─► convex hull ─► knee ─► ladder rungs
                                                              ▼
                       chunked parallel encode (split at scene cuts, one IDR per chunk, stitch)
                                                              ▼
                          rung_240p_crf26.h264 … rung_720p_crf18.h264  +  ladder.json
```

## Build and run

Dependencies: a C++20 compiler, CMake ≥ 3.20, and (optionally) FFmpeg's development libraries. The core library,
tests and benchmarks need nothing else; `-DFRAMEFORGE_WITH_FFMPEG=OFF` skips the libav layer.

```bash
# macOS
brew install cmake ninja pkg-config ffmpeg

# Ubuntu / Debian
sudo apt install build-essential cmake ninja-build pkg-config ffmpeg \
     libavformat-dev libavcodec-dev libavutil-dev libswscale-dev

cmake -S . -B build -G Ninja && cmake --build build
ctest --test-dir build --output-on-failure
```

```bash
tools/make_sample.sh                                   # 6 s synthetic 720p clip, three different scenes
build/frameforge cpu                                   # which SIMD backend is active
build/frameforge ladder samples/sample_720p.mp4 --frames 0 --out out --encode
build/frameforge compare samples/sample_720p.mp4       # BD-rate: libx264 vs libx265
build/frameforge analyze out/rung_720p_crf18.h264      # SPS, GOP structure, frame sizes
build/bench_luma && build/bench_resize && build/bench_pipeline
tools/check.sh all                                     # release, forced-scalar, TSan, ASan+UBSan, no-ffmpeg
```

## What it produces

On the sample clip (1280×720, 6 s, mandelbrot → gradients → test pattern with grain; 2 scene cuts found at frames 60
and 120). 24 probe encodes, 8 cores, Apple M1:

```
  bitrate ladder (6 rungs; 24 probes -> 13 on the Pareto frontier -> 5 on the convex hull)
    #1   428x240   crf 38       53 kbps   PSNR 33.39 dB   SSIM 0.9259
    #2   428x240   crf 26      118 kbps   PSNR 37.35 dB   SSIM 0.9468
    #3   428x240   crf 22      212 kbps   PSNR 38.47 dB   SSIM 0.9511
    #4   640x360   crf 22      520 kbps   PSNR 39.56 dB   SSIM 0.9615
    #5  1280x720   crf 26     1601 kbps   PSNR 42.63 dB   SSIM 0.9795
    #6  1280x720   crf 18     4417 kbps   PSNR 47.81 dB   SSIM 0.9875

  probe wall 5.9 s, cpu 42.8 s (7.3x parallel), 27 jobs, 0 retries
```

The resolution climbs with the bitrate: at 53–212 kbps this clip is better served by a sharp 240p than a starved
720p, which is exactly the decision a fixed ladder gets wrong. PSNR here is measured the way streaming
services do it: each rendition is decoded, **upscaled back to the source size**, and compared with the source, so
renditions of different resolution land on one quality axis.

## How the concepts map to code

| Area | Concept | Where |
|---|---|---|
| **Codecs** | Annex B NAL splitting, emulation prevention, SPS/PPS/slice-header parsing (incl. cropping, VUI colour + timing, scaling lists), GOP/IDR/frame-type statistics | [`h264_parser`](src/codec/h264_parser.cpp), `frameforge analyze` |
| | Send/receive libav API, encode → decode round trip with no disk I/O, raw Annex B decoding via the libav parser | [`encoder.cpp`](src/codec/encoder.cpp), [`decoder.cpp`](src/codec/decoder.cpp) |
| | CRF probing, closed GOPs, repeated SPS/PPS so chunks stand alone | [`chunked_encode.cpp`](src/optimizer/chunked_encode.cpp) |
| **Quality** | PSNR (SIMD SSD kernel), SSIM (8×8 windows from 4×4 block sums, one pass), BD-rate (cubic fit of log-rate vs quality) | [`src/quality`](src/quality) |
| **Detection** | Scene-cut detection: 4× box-downscaled luma, SIMD SAD, absolute + relative threshold so fast motion is not a cut | [`scene_cut.cpp`](src/pipeline/scene_cut.cpp) |
| **Enhancement** | Resize: 2×2 box pyramid (anti-aliasing) + fixed-point bilinear, I420-aware | [`resize.cpp`](src/kernels/resize.cpp) |
| **Framework** | Lock-free SPSC ring buffer: acquire/release ordering, cache-line separation, cached opposite index | [`ring_buffer.hpp`](include/pipeline/ring_buffer.hpp) |
| | Blocking `Channel` with close/cancel; bounded queues = backpressure | same |
| | Frame pool: preallocated 64-byte-aligned I420 frames, ownership moves instead of copying; an empty pool blocks the producer | [`buffer_pool.hpp`](include/pipeline/buffer_pool.hpp), [`frame.hpp`](include/pipeline/frame.hpp) |
| | Threaded source → stages → sink with per-stage busy time and stall counters | [`pipeline.cpp`](src/pipeline/pipeline.cpp) |
| **Performance** | NEON and AVX2 kernels (SAD, SSD, sum, row blend, 2×2 downscale), scalar reference, runtime dispatch, tails handled | [`src/kernels`](src/kernels) |
| | Bit-exact testing against the scalar reference; profile-driven resize rewrite (below) | [`test_kernels.cpp`](tests/kernels/test_kernels.cpp) |
| **Platform** | Job runner: thread pool, retries, idempotent jobs, stats (queue depth, retries, busy time) | [`job_runner.cpp`](src/optimizer/job_runner.cpp) |
| | Convex hull / Pareto frontier / knee-based rung selection | [`ladder.cpp`](src/optimizer/ladder.cpp) |
| | Split-and-stitch encoding with scene-cut-aware chunk planning | [`chunk_plan.cpp`](src/optimizer/chunk_plan.cpp) |
| | Cost model (cost per video minute, break-even views) | `cost_per_video_minute`, `break_even_views` |

## Design decisions worth explaining

**One codebase, two SIMD backends.** Each kernel exists three times: a scalar reference (compiled with
auto-vectorization *off*, so it is both the oracle and an honest baseline), a NEON version, and an AVX2 version.
`dispatch.cpp` picks one at startup (`__builtin_cpu_supports("avx2")` on x86; NEON is mandatory on AArch64).
`avx2.cpp` is the only file built with `-mavx2`, so the rest of the library still runs on any x86-64 CPU.
Tests run every available backend against the scalar one, over odd widths (0, 1, 7, 15, 17, 33…), strided blocks, and
extreme values, and require **bit-exact** output. `FRAMEFORGE_ISA=scalar` forces a backend.

**Rounding is part of the contract.** `blend_rows` is `(a·(256−w) + b·w + 128) >> 8`. NEON gets that from
`vrshrn`; AVX2 does it with 16-bit lanes, which cannot overflow because the weights sum to 256
(max 255·256 + 128 = 65 408). `w = 0` is special-cased because weight 256 does not fit in a byte.

**The ring buffer has two atomics and no CAS.** The producer publishes with a `release` store of `tail`; the consumer's
`acquire` load that sees it also sees the slot. Producer-owned and consumer-owned state sit on separate cache
lines, and each side caches the other's index so the fast path touches no shared line. With one producer and one
consumer there is no ABA problem to solve.

**Hull for the knee, frontier for the rungs.** The upper convex hull of (log₂ bitrate, quality) is the idealised
curve; its slope says where extra bits stop paying (default: < 0.75 dB per doubling). But with a coarse CRF grid the
hull throws away real, non-dominated encodes that sit slightly under a long segment, and the first version of this
code produced a ladder that jumped from 118 kbps to 4.4 Mbps. Rungs are now chosen from the **Pareto frontier**
below the knee, thinned evenly in log-bitrate if there are more than `--max-rungs`.

**Chunked encoding is only valid if chunks are independent.** Each chunk is its own encoder instance: one GOP, starts on an
IDR, nothing references across the boundary, SPS/PPS repeated, constant CRF so quality is consistent. The tests check
that the stitched stream has exactly one IDR and one SPS per chunk, that it is byte-identical on a second run
(jobs are idempotent, so a retry after a failed worker is safe), and that it decodes to the right frame count.
Independence costs bits: at 53 kbps the stitched 240p stream is ~18% larger than the monolithic probe, at 4.4 Mbps
it is ~1%. Short chunks hurt low-bitrate rungs most.

**Backpressure is real.** The frame pool has a fixed number of frames, so a slow downstream stage makes `acquire()`
block the source. A test pins 3 frames in a pipeline with a deliberately slow sink and asserts the source never has
more than 3 in flight. Teardown on error needs the source to be woken too, so `FramePipeline::run` takes an
`on_cancel` hook (usually `pool->close()`); without it a source blocked in `acquire()` would deadlock the join.
(That deadlock happened during development and is now a regression test.)

## Measurements

Apple M1, Release build, 1080p luma planes. Your machine will differ; run the benchmarks to fill in the AVX2 column.

| kernel | scalar GB/s | NEON GB/s | speedup |
|---|---:|---:|---:|
| SAD 1920×1080 | 4.39 | 48.31 | **11.0×** |
| SSD 1920×1080 | 3.60 | 21.22 | 5.9× |
| plane sum | 3.05 | 26.22 | 8.6× |
| SAD 16×16 blocks (motion-search shape) | 4.43 | 24.51 | 5.5× |

| case (per call) | scalar µs | NEON µs | speedup |
|---|---:|---:|---:|
| `blend_rows`, 1920-px row | 1.2 | 0.1 | 15.6× |
| `downscale2x` 1080p → 540p | 315.5 | 31.6 | 10.0× |
| `resize_plane` 1080p → 720p | 1429.9 | 673.8 | 2.1× |
| `resize_plane` 1080p → 360p (pyramid) | 675.6 | 203.2 | 3.3× |
| `resize_frame` I420 1080p → 720p | 2354.6 | 1031.4 | 2.3× |

| pipeline: decode-free 1080p, luma-mean → downscale | fps |
|---|---:|
| scalar kernels | ~1400 |
| NEON kernels | ~11 700 |

| SPSC queue, 2 threads | lock-free ring | mutex + condvar |
|---|---:|---:|
| capacity 4 | 22.6 M items/s | 1.0 M items/s |
| capacity 1024 | 35.7 M items/s | 16.0 M items/s |

Acquiring a pooled frame takes ~49 ns; allocating and zeroing a fresh 1080p frame takes ~141 µs.

**A profile-driven fix.** The first `resize_plane` did the horizontal pass first (a gather, so scalar) on two source rows
per output row, then SIMD-blended. SIMD made the whole call only 1.28× faster (1.95 ms) because the scalar pass was
~80% of the time: Amdahl's law in one benchmark. Doing the vertical blend first, full-width on the SIMD kernel, leaves
one row for the scalar pass: 1.95 ms → 0.67 ms. The horizontal gather is the next thing to vectorize.

## How it was verified

* **98 test cases in 11 suites** (~280 assertions, several looping over thousands of random inputs), run on every backend and again with `FRAMEFORGE_ISA=scalar`.
* **ThreadSanitizer** on all suites. As a negative control, weakening the ring buffer's release/acquire to relaxed makes
  TSan report a data race in `ring_spsc_stress_preserves_order_and_payload`; the shipped ordering is clean.
* **ASan + UBSan** on all suites.
* **Ground truth from outside the code base:** the mean per-frame luma PSNR that FrameForge reports for the stitched
  720p rung (47.83 dB) matches ffmpeg's `psnr` filter on the same two files (47.83 dB); `ffprobe` decodes the stitched
  stream to the expected 180 frames; `frameforge analyze` agrees on profile, resolution and GOP structure with the
  encoder's settings. The H.264 parser is also tested against hand-built SPS/PPS bitstreams (cropping 1088 → 1080,
  VUI timing 29.97 fps, scaling matrices, emulation-prevention bytes, truncated input) and real x264 output.
* **Property tests** for the hull (every probe lies on or below it), chunk planning (contiguous, covers every frame,
  length bounds), and ladder selection (strictly increasing, within caps) over thousands of random inputs.

## Limitations (read before quoting numbers)

* **The AVX2 backend has not been executed by me.** It compiles (checked with an x86_64 cross-build, AVX instructions
  present in the object code), but the machine this was developed on is an M1 and its Rosetta has no AVX2. The first
  thing to run on the Linux box is `tools/check.sh`; any AVX2 mistake shows up as a bit-exactness failure that names the
  kernel. Then run `build/bench_luma` to get the real AVX2 numbers.
* **PSNR and SSIM, not VMAF.** Rungs are placed on a PSNR curve; production systems use VMAF. This build's ffmpeg has
  `libvmaf`, so an `ffmpeg -lavfi libvmaf` cross-check of the chosen rungs is the obvious next step.
* **The sample clip is synthetic.** The `compare` result (x265 vs x264 at `medium`: +0.4% BD-rate on PSNR) says
  nothing general about codecs; on natural footage x265 is typically a large win. Use real clips for any claim.
* The clip is held in memory (default 90 frames; `--frames 0` loads everything), so this targets short clips and
  per-shot analysis rather than hour-long VOD. Streaming, per-shot ladders and a distributed worker pool are the
  design's natural extensions (the job runner already assumes idempotent tasks).
* H.264 and HEVC via libx264/libx265 only; Annex B in/out only (no MP4/HLS/DASH packaging); 8-bit 4:2:0; no audio.
* The horizontal resize pass is scalar.

## Layout

```
include/{kernels,pipeline,codec,quality,optimizer}   public headers
src/…                                                implementations (kernels/{scalar,neon,avx2}.cpp)
apps/frameforge.cpp                                  CLI: ladder | compare | analyze | cpu
tests/                                               dependency-free test harness + suites
benchmarks/                                          luma, resize, pipeline
tools/                                               make_sample.sh, check.sh
```
