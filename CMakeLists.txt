cmake_minimum_required(VERSION 3.20)
project(FrameForge VERSION 0.1.0
        DESCRIPTION "Media processing framework and per-title bitrate ladder optimizer"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(FRAMEFORGE_WITH_FFMPEG "Build the libav-based codec layer, probing and the CLI ladder command" ON)
option(FRAMEFORGE_BUILD_APPS "Build the frameforge CLI" ON)
option(FRAMEFORGE_BUILD_TESTS "Build unit tests" ON)
option(FRAMEFORGE_BUILD_BENCHMARKS "Build micro-benchmarks" ON)
set(FRAMEFORGE_SANITIZE "" CACHE STRING "Sanitizers, e.g. address,undefined or thread")

if(FRAMEFORGE_SANITIZE)
  add_compile_options(-fsanitize=${FRAMEFORGE_SANITIZE} -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=${FRAMEFORGE_SANITIZE})
  # UBSan's indirect-call 'function' check breaks C++ exception unwinding in Homebrew
  # clang 23 (every throw/catch pair aborts); all other UBSan checks are unaffected.
  if(FRAMEFORGE_SANITIZE MATCHES "undefined" AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-fno-sanitize=function)
  endif()
endif()

# ---------------------------------------------------------------- architecture
# Which SIMD backend gets compiled in. Runtime dispatch still checks the CPU.
if(CMAKE_OSX_ARCHITECTURES)
  set(FF_PROC "${CMAKE_OSX_ARCHITECTURES}")
else()
  set(FF_PROC "${CMAKE_SYSTEM_PROCESSOR}")
endif()
string(TOLOWER "${FF_PROC}" FF_PROC)
if(FF_PROC MATCHES "^(arm64|aarch64)")
  set(FF_ARCH arm64)
elseif(FF_PROC MATCHES "^(x86_64|amd64|x64)")
  set(FF_ARCH x86_64)
else()
  set(FF_ARCH generic)
endif()
message(STATUS "FrameForge: target arch = ${FF_ARCH}")

set(FF_WARNINGS -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)

# ---------------------------------------------------------------- core library
set(FF_CORE_SOURCES
  src/kernels/dispatch.cpp
  src/kernels/scalar.cpp
  src/kernels/resize.cpp
  src/kernels/yuv.cpp
  src/pipeline/frame.cpp
  src/pipeline/pipeline.cpp
  src/pipeline/scene_cut.cpp
  src/quality/psnr.cpp
  src/quality/ssim.cpp
  src/quality/bdrate.cpp
  src/codec/h264_parser.cpp
  src/optimizer/ladder.cpp
  src/optimizer/job_runner.cpp
  src/optimizer/chunk_plan.cpp)

if(FF_ARCH STREQUAL "arm64")
  list(APPEND FF_CORE_SOURCES src/kernels/neon.cpp)
elseif(FF_ARCH STREQUAL "x86_64")
  list(APPEND FF_CORE_SOURCES src/kernels/avx2.cpp)
  set_source_files_properties(src/kernels/avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2")
endif()

# The scalar kernels are the bit-exact reference *and* the "no SIMD" benchmark
# baseline, so keep the compiler from auto-vectorizing them.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set_source_files_properties(src/kernels/scalar.cpp PROPERTIES
    COMPILE_OPTIONS "-fno-vectorize;-fno-slp-vectorize")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set_source_files_properties(src/kernels/scalar.cpp PROPERTIES
    COMPILE_OPTIONS "-fno-tree-vectorize;-fno-tree-slp-vectorize")
endif()

find_package(Threads REQUIRED)
add_library(frameforge_core STATIC ${FF_CORE_SOURCES})
target_include_directories(frameforge_core PUBLIC include)
target_link_libraries(frameforge_core PUBLIC Threads::Threads)
target_compile_options(frameforge_core PRIVATE ${FF_WARNINGS})
if(FF_ARCH STREQUAL "arm64")
  target_compile_definitions(frameforge_core PRIVATE FF_HAVE_NEON=1)
elseif(FF_ARCH STREQUAL "x86_64")
  target_compile_definitions(frameforge_core PRIVATE FF_HAVE_AVX2=1)
endif()

# ---------------------------------------------------------------- codec layer
if(FRAMEFORGE_WITH_FFMPEG)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat libavcodec libavutil libswscale)
  add_library(frameforge_codec STATIC
    src/codec/decoder.cpp
    src/codec/encoder.cpp
    src/optimizer/probe.cpp
    src/optimizer/chunked_encode.cpp)
  target_link_libraries(frameforge_codec PUBLIC frameforge_core PkgConfig::FFMPEG)
  target_compile_definitions(frameforge_codec PUBLIC FF_WITH_FFMPEG=1)
  target_compile_options(frameforge_codec PRIVATE ${FF_WARNINGS})
endif()

# ---------------------------------------------------------------- app
if(FRAMEFORGE_BUILD_APPS)
  add_executable(frameforge apps/frameforge.cpp)
  target_compile_options(frameforge PRIVATE ${FF_WARNINGS})
  # frameforge_codec already exports frameforge_core
  target_link_libraries(frameforge PRIVATE $<IF:$<BOOL:${FRAMEFORGE_WITH_FFMPEG}>,frameforge_codec,frameforge_core>)
endif()

# ---------------------------------------------------------------- tests
if(FRAMEFORGE_BUILD_TESTS)
  enable_testing()
  function(ff_add_test name)
    add_executable(${name} ${ARGN} tests/test_main.cpp)
    target_include_directories(${name} PRIVATE tests)
    target_compile_options(${name} PRIVATE ${FF_WARNINGS})
    if(FRAMEFORGE_WITH_FFMPEG)
      target_link_libraries(${name} PRIVATE frameforge_codec)  # exports frameforge_core
    else()
      target_link_libraries(${name} PRIVATE frameforge_core)
    endif()
    add_test(NAME ${name} COMMAND ${name})
  endfunction()

  ff_add_test(test_kernels   tests/kernels/test_kernels.cpp)
  ff_add_test(test_resize    tests/kernels/test_resize.cpp)
  ff_add_test(test_ring      tests/pipeline/test_ring_buffer.cpp)
  ff_add_test(test_pool      tests/pipeline/test_buffer_pool.cpp)
  ff_add_test(test_pipeline  tests/pipeline/test_pipeline.cpp)
  ff_add_test(test_quality   tests/quality/test_quality.cpp)
  ff_add_test(test_ladder    tests/optimizer/test_ladder.cpp)
  ff_add_test(test_jobs      tests/optimizer/test_jobs.cpp)
  ff_add_test(test_scenecut  tests/optimizer/test_scene_cut.cpp)
  ff_add_test(test_h264      tests/codec/test_h264_parser.cpp)
  if(FRAMEFORGE_WITH_FFMPEG)
    ff_add_test(test_codec   tests/codec/test_codec.cpp)
  endif()
endif()

# ---------------------------------------------------------------- benchmarks
if(FRAMEFORGE_BUILD_BENCHMARKS)
  foreach(b luma resize pipeline)
    add_executable(bench_${b} benchmarks/${b}.cpp)
    target_include_directories(bench_${b} PRIVATE benchmarks)
    target_link_libraries(bench_${b} PRIVATE frameforge_core)
    target_compile_options(bench_${b} PRIVATE ${FF_WARNINGS})
  endforeach()
endif()
