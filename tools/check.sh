#!/usr/bin/env bash
# Build and test FrameForge. Works the same on macOS (arm64) and Linux (x86_64 / arm64).
#
#   tools/check.sh            release build + tests, with the SIMD backends the CPU supports
#   tools/check.sh scalar     same tests with SIMD forced off (FRAMEFORGE_ISA=scalar)
#   tools/check.sh tsan       ThreadSanitizer build  (lock-free ring buffer, pool, pipeline, job runner)
#   tools/check.sh asan       AddressSanitizer + UBSan build
#   tools/check.sh noffmpeg   core library only, no libav needed
#   tools/check.sh all        everything above
set -euo pipefail
cd "$(dirname "$0")/.."

jobs="$( (nproc 2>/dev/null || sysctl -n hw.ncpu) )"
gen=(); command -v ninja >/dev/null && gen=(-G Ninja)

run() {  # name, cmake args...
  local name="$1"; shift
  echo "=== $name ==="
  cmake -S . -B "build-$name" "${gen[@]}" "$@" >/dev/null
  cmake --build "build-$name" -j "$jobs" 2>&1 | grep -E "error|warning: " -A3 | grep -v "ignoring duplicate" || true
  ctest --test-dir "build-$name" --timeout 300 --output-on-failure | tail -n 5
}

case "${1:-release}" in
  release)  run release -DCMAKE_BUILD_TYPE=Release
            build-release/frameforge cpu ;;
  scalar)   run scalar -DCMAKE_BUILD_TYPE=Release -DFRAMEFORGE_BUILD_BENCHMARKS=OFF
            FRAMEFORGE_ISA=scalar ctest --test-dir build-scalar --timeout 300 | tail -n 4 ;;
  tsan)     run tsan -DCMAKE_BUILD_TYPE=RelWithDebInfo -DFRAMEFORGE_SANITIZE=thread -DFRAMEFORGE_BUILD_BENCHMARKS=OFF ;;
  asan)     run asan -DCMAKE_BUILD_TYPE=RelWithDebInfo -DFRAMEFORGE_SANITIZE="address,undefined" -DFRAMEFORGE_BUILD_BENCHMARKS=OFF ;;
  noffmpeg) run noffmpeg -DCMAKE_BUILD_TYPE=Release -DFRAMEFORGE_WITH_FFMPEG=OFF -DFRAMEFORGE_BUILD_BENCHMARKS=OFF ;;
  all)      for m in release scalar tsan asan noffmpeg; do "$0" "$m"; done ;;
  *)        sed -n '2,10p' "$0"; exit 2 ;;
esac
