cmake_minimum_required(VERSION 3.24)
project(tinyinfer_host_codec LANGUAGES CXX)
# Independent host-only build. No Python headers, Torch, CUDA, OpenMP or download.
add_library(tinyinfer_host_codec SHARED ../host_codec.cpp)
target_include_directories(tinyinfer_host_codec PUBLIC ../include)
target_compile_features(tinyinfer_host_codec PUBLIC cxx_std_17)
target_compile_options(tinyinfer_host_codec PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math)
