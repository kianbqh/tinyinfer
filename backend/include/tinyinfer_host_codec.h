#ifndef TINYINFER_HOST_CODEC_H
#define TINYINFER_HOST_CODEC_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
enum ti_host_status { TI_HOST_OK=0, TI_HOST_INVALID=1, TI_HOST_INTERNAL=2 };
int ti_host_codec_abi_version(void); /* v1; does not change the CUDA/backend ABI */
/* CPU-only synchronous host memory. No allocation, CUDA or mutable global state.
 * Extents are element counts. Alignment, byte-range overflow, pairwise overlap,
 * finite values and FE_TONEAREST are checked before output writes. Caller owns
 * actual accessible buffer extents: a pointer cannot establish allocation size.
 * f16_bits are already rounded IEEE binary16 [K,N], N=columns; q is [-127,127].
 * Validation failures leave outputs unchanged. Existing kernel ABI is unchanged. */
int ti_host_quantize_f16(const uint16_t* f16_bits, size_t elements, size_t columns,
                         int8_t* q, size_t q_elements, float* scales, size_t scale_elements);
/* FP32 multiplication followed by binary16 round-to-nearest, ties-to-even.
 * Rejects -128, nonfinite/nonpositive scales and finite values overflowing half. */
int ti_host_dequantize_i8(const int8_t* q, size_t elements, size_t columns,
                         const float* scales, size_t scale_elements,
                         uint16_t* f16_bits, size_t output_elements);
#ifdef __cplusplus
}
#endif
#endif
