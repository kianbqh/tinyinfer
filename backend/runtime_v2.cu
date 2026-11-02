#include "tinyinfer_backend_v2.h"
#include "tc/launchers.cuh"
#include "tc/catalog.cuh"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <limits>
namespace {
thread_local char error_text[256]="";
int fail(int status,const char* text) { std::snprintf(error_text,sizeof(error_text),"%s",text);return status; }
int cuda_status(cudaError_t status) {
  return status==cudaSuccess?TI_V2_OK:fail(TI_V2_CUDA_ERROR,cudaGetErrorString(status));
}
bool aligned(const void* p,size_t bytes) { return p && reinterpret_cast<uintptr_t>(p)%bytes==0; }
bool shape_fits(std::initializer_list<int> dims,size_t bytes) {
  uint64_t product=bytes;
  for(int d:dims) {
    if(d<=0 || product>uint64_t(std::numeric_limits<int64_t>::max())/uint64_t(d)) return false;
    product*=uint64_t(d);
  }
  return true;
}
int begin(int device) {
  error_text[0]='\0';
  if(device<0) return fail(TI_V2_INVALID_ARGUMENT,"device must be nonnegative");
  int current=-1;int status=cuda_status(cudaGetDevice(&current));
  if(status) return status;
  return current==device?TI_V2_OK:fail(TI_V2_DEVICE_MISMATCH,"caller must select the supplied CUDA device");
}
int check_gemm(const void* a,const void* b,const void* bias,void* y,int m,int n,int k,int dtype,int config,int device,bool int8) {
  int status=begin(device);if(status) return status;
  if(!ti::v2::config_available(config)) return fail(TI_V2_UNSUPPORTED_CONFIG,"GEMM config absent from this compiled allow-list");
  int bm=ti::v2::config_tile(config)==0?32:64;
  if((dtype!=0 && dtype!=1) || !aligned(a,2) || !aligned(b,int8?1:2) || !aligned(y,dtype==0?2:4) ||
     (bias && !aligned(bias,2)) || !shape_fits({m,k},2) || !shape_fits({k,n},int8?1:2) ||
     !shape_fits({m,n},dtype==0?2:4) || (m-1)/bm+1>65535)
    return fail(TI_V2_INVALID_ARGUMENT,"invalid GEMM shape, pointers, alignment, output dtype or grid");
  return TI_V2_OK;
}
}
extern "C" int ti_v2_abi_version(void) { return 2; }
extern "C" const char* ti_v2_last_error(void) { return error_text; }
extern "C" const char* ti_v2_gemm_catalog_json(void) {
  return "{\"schema_version\":1,\"only_config\":"
    #define TI_STRING_IMPL(x) #x
    #define TI_STRING(x) TI_STRING_IMPL(x)
    TI_STRING(TI_V2_ONLY_GEMM_CONFIG)
    #undef TI_STRING
    #undef TI_STRING_IMPL
    ",\"configs\":["
    "{\"id\":0,\"tile\":[32,32,16,4],\"stages\":2,\"epilogue\":\"fused\"},"
    "{\"id\":1,\"tile\":[64,32,32,8],\"stages\":2,\"epilogue\":\"fused\"},"
    "{\"id\":2,\"tile\":[64,64,32,8],\"stages\":2,\"epilogue\":\"fused\"},"
    "{\"id\":3,\"tile\":[32,32,16,4],\"stages\":1,\"epilogue\":\"fused\"},"
    "{\"id\":4,\"tile\":[64,32,32,8],\"stages\":1,\"epilogue\":\"fused\"},"
    "{\"id\":5,\"tile\":[64,64,32,8],\"stages\":1,\"epilogue\":\"fused\"},"
    "{\"id\":6,\"tile\":[32,32,16,4],\"stages\":2,\"epilogue\":\"separate\"},"
    "{\"id\":7,\"tile\":[64,32,32,8],\"stages\":2,\"epilogue\":\"separate\"},"
    "{\"id\":8,\"tile\":[64,64,32,8],\"stages\":2,\"epilogue\":\"separate\"},"
    "{\"id\":9,\"tile\":[32,32,16,4],\"stages\":1,\"epilogue\":\"separate\"},"
    "{\"id\":10,\"tile\":[64,32,32,8],\"stages\":1,\"epilogue\":\"separate\"},"
    "{\"id\":11,\"tile\":[64,64,32,8],\"stages\":1,\"epilogue\":\"separate\"}]}";
}
extern "C" int ti_v2_gemm_f16(const uint16_t* a,const uint16_t* b,const uint16_t* bias,void* y,
                              int m,int n,int k,int dtype,int config,uintptr_t stream,int device) {
  return ti_v2_gemm_ex_f16(a,b,bias,y,nullptr,m,n,k,dtype,config,stream,device);
}
extern "C" int ti_v2_gemm_i8(const uint16_t* a,const int8_t* b,const float* scales,const uint16_t* bias,void* y,
                             int m,int n,int k,int dtype,int config,uintptr_t stream,int device) {
  return ti_v2_gemm_ex_i8(a,b,scales,bias,y,nullptr,m,n,k,dtype,config,stream,device);
}
extern "C" int ti_v2_gemm_ex_f16(const uint16_t* a,const uint16_t* b,const uint16_t* bias,void* y,float* workspace,
                                 int m,int n,int k,int dtype,int config,uintptr_t stream,int device) {
  int status=check_gemm(a,b,bias,y,m,n,k,dtype,config,device,false);if(status) return status;
  if(ti::v2::config_separate(config) && (!aligned(workspace,4) || !shape_fits({m,n},4)))
    return fail(TI_V2_INVALID_ARGUMENT,"separate epilogue requires caller-owned FP32[M,N] workspace");
  return cuda_status(ti::v2::gemm(reinterpret_cast<const half*>(a),b,nullptr,reinterpret_cast<const half*>(bias),y,workspace,
                                 m,n,k,dtype,config,false,reinterpret_cast<cudaStream_t>(stream)));
}
extern "C" int ti_v2_gemm_ex_i8(const uint16_t* a,const int8_t* b,const float* scales,const uint16_t* bias,void* y,float* workspace,
                                int m,int n,int k,int dtype,int config,uintptr_t stream,int device) {
  int status=check_gemm(a,b,bias,y,m,n,k,dtype,config,device,true);if(status) return status;
  if(!aligned(scales,4)) return fail(TI_V2_INVALID_ARGUMENT,"scales pointer must be nonnull aligned FP32[N]");
  if(ti::v2::config_separate(config) && (!aligned(workspace,4) || !shape_fits({m,n},4)))
    return fail(TI_V2_INVALID_ARGUMENT,"separate epilogue requires caller-owned FP32[M,N] workspace");
  return cuda_status(ti::v2::gemm(reinterpret_cast<const half*>(a),b,scales,reinterpret_cast<const half*>(bias),y,workspace,
                                 m,n,k,dtype,config,true,reinterpret_cast<cudaStream_t>(stream)));
}
extern "C" int ti_v2_legacy_gemm_f32(const float* a,const float* b,float* y,int m,int n,int k,uintptr_t stream,int device) {
  int status=begin(device);if(status) return status;
  if(!aligned(a,4) || !aligned(b,4) || !aligned(y,4) || !shape_fits({m,k},4) || !shape_fits({k,n},4) ||
     !shape_fits({m,n},4) || (m-1)/64+1>65535) return fail(TI_V2_INVALID_ARGUMENT,"invalid FP32 GEMM arguments");
  return cuda_status(ti::v2::legacy_gemm(a,b,y,m,n,k,reinterpret_cast<cudaStream_t>(stream)));
}
extern "C" int ti_v2_attention_f16(const uint16_t* q,const uint16_t* k,const uint16_t* v,uint16_t* y,
                                   const int32_t* ql,const int32_t* kl,const int32_t* offsets,
                                   int b,int h,int nq,int nk,int d,int causal,int config,uintptr_t stream,int device) {
  int status=begin(device);if(status) return status;
  if(config<0 || config>1) return fail(TI_V2_UNSUPPORTED_CONFIG,"attention config must be 0,1");
  if((d!=64 && d!=128) || (causal!=0 && causal!=1) || b>65535 || h>65535 ||
     !shape_fits({b,h,nq,d},2) || !shape_fits({b,h,nk,d},2) ||
     !aligned(q,2) || !aligned(k,2) || !aligned(v,2) || !aligned(y,2) ||
     !aligned(ql,4) || !aligned(kl,4) || !aligned(offsets,4)) return fail(TI_V2_INVALID_ARGUMENT,"invalid attention arguments");
  return cuda_status(ti::v2::attention(reinterpret_cast<const half*>(q),reinterpret_cast<const half*>(k),
                                      reinterpret_cast<const half*>(v),reinterpret_cast<half*>(y),ql,kl,offsets,
                                      b,h,nq,nk,d,causal!=0,config,reinterpret_cast<cudaStream_t>(stream)));
}
extern "C" int ti_v2_layernorm_f16(const uint16_t* x,const uint16_t* g,const uint16_t* b,uint16_t* y,
                                   int rows,int cols,float eps,uintptr_t stream,int device) {
  int status=begin(device);if(status) return status;
  if(!shape_fits({rows,cols},2) || !std::isfinite(eps) || eps<=0.f || !aligned(x,2) || !aligned(g,2) ||
     !aligned(b,2) || !aligned(y,2)) return fail(TI_V2_INVALID_ARGUMENT,"invalid LayerNorm arguments");
  return cuda_status(ti::v2::layernorm(reinterpret_cast<const half*>(x),reinterpret_cast<const half*>(g),
                                      reinterpret_cast<const half*>(b),reinterpret_cast<half*>(y),rows,cols,eps,reinterpret_cast<cudaStream_t>(stream)));
}
extern "C" int ti_v2_gelu_new_f16(const uint16_t* x,uint16_t* y,int64_t count,uintptr_t stream,int device) {
  int status=begin(device);if(status) return status;
  if(count<=0 || count>std::numeric_limits<int64_t>::max()/2 || !aligned(x,2) || !aligned(y,2))
    return fail(TI_V2_INVALID_ARGUMENT,"invalid GELU arguments");
  return cuda_status(ti::v2::gelu_new(reinterpret_cast<const half*>(x),reinterpret_cast<half*>(y),count,reinterpret_cast<cudaStream_t>(stream)));
}
extern "C" int ti_v2_softmax_f16(const uint16_t* x,uint16_t* y,int rows,int cols,uintptr_t stream,int device) {
  int status=begin(device);if(status) return status;
  if(!shape_fits({rows,cols},2) || !aligned(x,2) || !aligned(y,2)) return fail(TI_V2_INVALID_ARGUMENT,"invalid softmax arguments");
  return cuda_status(ti::v2::softmax(reinterpret_cast<const half*>(x),reinterpret_cast<half*>(y),rows,cols,reinterpret_cast<cudaStream_t>(stream)));
}
