#include "tinyinfer_backend_v2.h"
#include "tinyinfer_host_validation.hpp"
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
int host_check(ti::host::Check check) {
  return check.status?fail(check.status,check.message):TI_V2_OK;
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
  return host_check(ti::host::gemm(a,b,bias,y,m,n,k,dtype,config,int8));
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
  status=host_check(ti::host::workspace(workspace,m,n,config));if(status) return status;
  return cuda_status(ti::v2::gemm(reinterpret_cast<const half*>(a),b,nullptr,reinterpret_cast<const half*>(bias),y,workspace,
                                 m,n,k,dtype,config,false,reinterpret_cast<cudaStream_t>(stream)));
}
extern "C" int ti_v2_gemm_ex_i8(const uint16_t* a,const int8_t* b,const float* scales,const uint16_t* bias,void* y,float* workspace,
                                int m,int n,int k,int dtype,int config,uintptr_t stream,int device) {
  int status=check_gemm(a,b,bias,y,m,n,k,dtype,config,device,true);if(status) return status;
  status=host_check(ti::host::scales(scales));if(status) return status;
  status=host_check(ti::host::workspace(workspace,m,n,config));if(status) return status;
  return cuda_status(ti::v2::gemm(reinterpret_cast<const half*>(a),b,scales,reinterpret_cast<const half*>(bias),y,workspace,
                                 m,n,k,dtype,config,true,reinterpret_cast<cudaStream_t>(stream)));
}
extern "C" int ti_v2_legacy_gemm_f32(const float* a,const float* b,float* y,int m,int n,int k,uintptr_t stream,int device) {
  int status=begin(device);if(status) return status;
  status=host_check(ti::host::legacy_gemm(a,b,y,m,n,k));if(status) return status;
  return cuda_status(ti::v2::legacy_gemm(a,b,y,m,n,k,reinterpret_cast<cudaStream_t>(stream)));
}
extern "C" int ti_v2_attention_f16(const uint16_t* q,const uint16_t* k,const uint16_t* v,uint16_t* y,
                                   const int32_t* ql,const int32_t* kl,const int32_t* offsets,
                                   int b,int h,int nq,int nk,int d,int causal,int config,uintptr_t stream,int device) {
  int status=begin(device);if(status) return status;
  status=host_check(ti::host::attention(q,k,v,y,ql,kl,offsets,b,h,nq,nk,d,causal,config));if(status) return status;
  return cuda_status(ti::v2::attention(reinterpret_cast<const half*>(q),reinterpret_cast<const half*>(k),
                                      reinterpret_cast<const half*>(v),reinterpret_cast<half*>(y),ql,kl,offsets,
                                      b,h,nq,nk,d,causal!=0,config,reinterpret_cast<cudaStream_t>(stream)));
}
extern "C" int ti_v2_layernorm_f16(const uint16_t* x,const uint16_t* g,const uint16_t* b,uint16_t* y,
                                   int rows,int cols,float eps,uintptr_t stream,int device) {
  int status=begin(device);if(status) return status;
  status=host_check(ti::host::layernorm(x,g,b,y,rows,cols,eps));if(status) return status;
  return cuda_status(ti::v2::layernorm(reinterpret_cast<const half*>(x),reinterpret_cast<const half*>(g),
                                      reinterpret_cast<const half*>(b),reinterpret_cast<half*>(y),rows,cols,eps,reinterpret_cast<cudaStream_t>(stream)));
}
extern "C" int ti_v2_gelu_new_f16(const uint16_t* x,uint16_t* y,int64_t count,uintptr_t stream,int device) {
  int status=begin(device);if(status) return status;
  status=host_check(ti::host::gelu(x,y,count));if(status) return status;
  return cuda_status(ti::v2::gelu_new(reinterpret_cast<const half*>(x),reinterpret_cast<half*>(y),count,reinterpret_cast<cudaStream_t>(stream)));
}
extern "C" int ti_v2_softmax_f16(const uint16_t* x,uint16_t* y,int rows,int cols,uintptr_t stream,int device) {
  int status=begin(device);if(status) return status;
  status=host_check(ti::host::softmax(x,y,rows,cols));if(status) return status;
  return cuda_status(ti::v2::softmax(reinterpret_cast<const half*>(x),reinterpret_cast<half*>(y),rows,cols,reinterpret_cast<cudaStream_t>(stream)));
}
