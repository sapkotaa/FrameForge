#!/usr/bin/env bash
# Generates a 6-second 720p30 test clip with three very different scenes, so the
# ladder optimizer and scene-cut detector have something to chew on:
#   0-2 s  mandelbrot zoom   (fine detail, high complexity)
#   2-4 s  smooth gradients  (cheap to encode)
#   4-6 s  test pattern + light grain (moderately hard)
# Needs ffmpeg with lavfi. Usage: tools/make_sample.sh [output.mp4]
set -euo pipefail
out="${1:-samples/sample_720p.mp4}"
mkdir -p "$(dirname "$out")"
ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i "mandelbrot=size=1280x720:rate=30:end_scale=0.01" \
  -f lavfi -i "gradients=size=1280x720:rate=30:speed=0.02:nb_colors=4" \
  -f lavfi -i "testsrc2=size=1280x720:rate=30,noise=alls=3:allf=t" \
  -filter_complex "[0:v]trim=duration=2,setpts=PTS-STARTPTS,format=yuv420p[a];[1:v]trim=duration=2,setpts=PTS-STARTPTS,format=yuv420p[b];[2:v]trim=duration=2,setpts=PTS-STARTPTS,format=yuv420p[c];[a][b][c]concat=n=3:v=1:a=0[v]" \
  -map "[v]" -c:v libx264 -preset fast -crf 12 -pix_fmt yuv420p -r 30 "$out"
ffprobe -v error -show_entries stream=width,height,r_frame_rate,nb_frames -of default=nw=1 "$out"
