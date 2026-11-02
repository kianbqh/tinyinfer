#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
namespace ti::v2 {
cudaError_t gemm(const half*, const void*, const float*, const half*, void*, float*,int,int,int,int,int,bool,cudaStream_t);
cudaError_t legacy_gemm(const float*,const float*,float*,int,int,int,cudaStream_t);
cudaError_t attention(const half*,const half*,const half*,half*,const int32_t*,const int32_t*,const int32_t*,int,int,int,int,int,bool,int,cudaStream_t);
cudaError_t layernorm(const half*,const half*,const half*,half*,int,int,float,cudaStream_t);
cudaError_t gelu_new(const half*,half*,int64_t,cudaStream_t);
cudaError_t softmax(const half*,half*,int,int,cudaStream_t);
}
