#ifndef TINYINFER_BACKEND_V2_H
#define TINYINFER_BACKEND_V2_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
enum ti_v2_status { TI_V2_OK=0, TI_V2_INVALID_ARGUMENT=1, TI_V2_DEVICE_MISMATCH=2,
                    TI_V2_UNSUPPORTED_CONFIG=3, TI_V2_CUDA_ERROR=4 };
enum ti_v2_output_dtype { TI_V2_FP16=0, TI_V2_FP32=1 };
int ti_v2_abi_version(void);
const char* ti_v2_last_error(void);
const char* ti_v2_gemm_catalog_json(void);
/* All pointers device-resident, contiguous row-major; stream/device/lifetime owned by caller.
 * Inputs finite, outputs nonaliasing. Dimension order M,N,K, unlike the legacy ABI.
 * output_dtype 0=FP16 bits, 1=FP32. Bias nullable FP16[N]. See INTERFACES.md. */
int ti_v2_gemm_f16(const uint16_t*,const uint16_t*,const uint16_t*,void*,
                   int m,int n,int k,int output_dtype,int config,uintptr_t stream,int device);
int ti_v2_gemm_i8(const uint16_t*,const int8_t*,const float*,const uint16_t*,void*,
                  int m,int n,int k,int output_dtype,int config,uintptr_t stream,int device);
/* _ex adds caller-owned aligned FP32[M,N] workspace for separate epilogue configs.
 * Workspace must not alias any other allocation. Null for fused epilogue configs.
 * Old exports above support fused epilogues only; configs0/1/2 remain unchanged. */
int ti_v2_gemm_ex_f16(const uint16_t*,const uint16_t*,const uint16_t*,void*,float* workspace,
                      int m,int n,int k,int output_dtype,int config,uintptr_t stream,int device);
int ti_v2_gemm_ex_i8(const uint16_t*,const int8_t*,const float*,const uint16_t*,void*,float* workspace,
                     int m,int n,int k,int output_dtype,int config,uintptr_t stream,int device);
int ti_v2_legacy_gemm_f32(const float*,const float*,float*,int m,int n,int k,
                          uintptr_t stream,int device);
/* q[B,H,Q,D], k/v[B,H,K,D], metadata int32[B]. All-masked/padded output rows=0.
 * Valid length ranges [0,Q],[0,K]; offset [0,INT32_MAX-Q]. D=64/128. */
int ti_v2_attention_f16(const uint16_t*,const uint16_t*,const uint16_t*,uint16_t*,
                        const int32_t* q_lengths,const int32_t* kv_lengths,
                        const int32_t* query_offsets,int batch,int heads,int q_len,
                        int kv_len,int head_dim,int causal,int config,uintptr_t stream,int device);
int ti_v2_layernorm_f16(const uint16_t*,const uint16_t*,const uint16_t*,uint16_t*,
                        int rows,int cols,float eps,uintptr_t stream,int device);
int ti_v2_gelu_new_f16(const uint16_t*,uint16_t*,int64_t count,uintptr_t stream,int device);
int ti_v2_softmax_f16(const uint16_t*,uint16_t*,int rows,int cols,uintptr_t stream,int device);
#ifdef __cplusplus
}
#endif
#endif
