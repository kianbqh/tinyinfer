#pragma once
// Production scalar operations shared by the host codec and CUDA weight load.
#include <cstdint>
#if defined(__CUDACC__)
#define TI_HOST_DEVICE __host__ __device__
#else
#define TI_HOST_DEVICE
#endif
namespace ti::host {
TI_HOST_DEVICE inline float dequantize_scalar(int8_t value, float scale) {
  return static_cast<float>(value) * scale;
}
inline bool finite_half_bits(uint16_t value) { return (value & 0x7c00u) != 0x7c00u; }
}  // namespace ti::host
#undef TI_HOST_DEVICE
