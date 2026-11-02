#pragma once
// Production metadata checks: no CUDA calls, pointer dereferences or allocations.
// The C ABI invokes these same routines after its existing device check.
#include "tinyinfer_backend_v2.h"
#include "../../kernels/src/cuda/tc/catalog.cuh"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
namespace ti::host {
struct Check { int status=TI_V2_OK; const char* message=""; };
inline Check invalid(const char* text) { return {TI_V2_INVALID_ARGUMENT,text}; }
inline bool aligned(const void* p,size_t bytes) { return p && reinterpret_cast<uintptr_t>(p)%bytes==0; }
inline bool shape_fits(std::initializer_list<int> dims,size_t bytes) {
  uint64_t product=bytes;
  for(int d:dims) {
    if(d<=0 || product>uint64_t(std::numeric_limits<int64_t>::max())/uint64_t(d)) return false;
    product*=uint64_t(d);
  }
  return true;
}
inline Check gemm(const void* a,const void* b,const void* bias,void* y,
                   int m,int n,int k,int dtype,int config,bool int8) {
  if(!ti::v2::config_available(config))
    return {TI_V2_UNSUPPORTED_CONFIG,"GEMM config absent from this compiled allow-list"};
  const int bm=ti::v2::config_tile(config)==0?32:64;
  if((dtype!=0 && dtype!=1) || !aligned(a,2) || !aligned(b,int8?1:2) || !aligned(y,dtype==0?2:4) ||
     (bias && !aligned(bias,2)) || !shape_fits({m,k},2) || !shape_fits({k,n},int8?1:2) ||
     !shape_fits({m,n},dtype==0?2:4) || (m-1)/bm+1>65535)
    return invalid("invalid GEMM shape, pointers, alignment, output dtype or grid");
  return {};
}
inline Check workspace(void* p,int m,int n,int config) {
  if(ti::v2::config_separate(config) && (!aligned(p,4) || !shape_fits({m,n},4)))
    return invalid("separate epilogue requires caller-owned FP32[M,N] workspace");
  return {};
}
inline Check scales(const void* p) {
  return aligned(p,4)?Check{}:invalid("scales pointer must be nonnull aligned FP32[N]");
}
inline Check legacy_gemm(const void* a,const void* b,void* y,int m,int n,int k) {
  if(!aligned(a,4) || !aligned(b,4) || !aligned(y,4) || !shape_fits({m,k},4) || !shape_fits({k,n},4) ||
     !shape_fits({m,n},4) || (m-1)/64+1>65535) return invalid("invalid FP32 GEMM arguments");
  return {};
}
inline Check attention(const void* q,const void* k,const void* v,void* y,
                       const void* ql,const void* kl,const void* offsets,
                       int b,int h,int nq,int nk,int d,int causal,int config) {
  if(config<0 || config>1) return {TI_V2_UNSUPPORTED_CONFIG,"attention config must be 0,1"};
  if((d!=64 && d!=128) || (causal!=0 && causal!=1) || b>65535 || h>65535 ||
     !shape_fits({b,h,nq,d},2) || !shape_fits({b,h,nk,d},2) ||
     !aligned(q,2) || !aligned(k,2) || !aligned(v,2) || !aligned(y,2) ||
     !aligned(ql,4) || !aligned(kl,4) || !aligned(offsets,4)) return invalid("invalid attention arguments");
  return {};
}
inline Check layernorm(const void* x,const void* g,const void* b,void* y,int rows,int cols,float eps) {
  if(!shape_fits({rows,cols},2) || !std::isfinite(eps) || eps<=0.f || !aligned(x,2) || !aligned(g,2) ||
     !aligned(b,2) || !aligned(y,2)) return invalid("invalid LayerNorm arguments");
  return {};
}
inline Check gelu(const void* x,void* y,int64_t count) {
  if(count<=0 || count>std::numeric_limits<int64_t>::max()/2 || !aligned(x,2) || !aligned(y,2))
    return invalid("invalid GELU arguments");
  return {};
}
inline Check softmax(const void* x,void* y,int rows,int cols) {
  if(!shape_fits({rows,cols},2) || !aligned(x,2) || !aligned(y,2)) return invalid("invalid softmax arguments");
  return {};
}
} // namespace ti::host
