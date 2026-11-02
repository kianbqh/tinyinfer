#pragma once
#ifndef TI_V2_ONLY_GEMM_CONFIG
#define TI_V2_ONLY_GEMM_CONFIG -1
#endif
#if TI_V2_ONLY_GEMM_CONFIG < -1 || TI_V2_ONLY_GEMM_CONFIG > 11
#error "TI_V2_ONLY_GEMM_CONFIG must be -1 or an allow-listed ID0..11"
#endif
namespace ti::v2 {
inline constexpr bool config_available(int id) {
  return id>=0 && id<12 && (TI_V2_ONLY_GEMM_CONFIG==-1 || TI_V2_ONLY_GEMM_CONFIG==id);
}
inline constexpr int config_tile(int id) { return id%3; }
inline constexpr int config_stages(int id) { return (id>=3 && id<6) || id>=9?1:2; }
inline constexpr bool config_separate(int id) { return id>=6; }
}
