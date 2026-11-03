#include "tinyinfer_host_validation.hpp"
#include "input.hpp"
#include <array>
#include <cstdlib>
namespace {
volatile int sink=0;
void consume(ti::host::Check x){if(x.status<0 || x.status>TI_V2_UNSUPPORTED_CONFIG || !x.message)std::abort();sink=x.status;}
}
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data,size_t size) {
  if(size>ti_fuzz::max_input)return 0;
  ti_fuzz::Input in{data,size};
  // Every pointer is null or into live local storage. Validators never dereference.
  alignas(64) std::array<uint8_t,256> storage{};
  auto ptr=[&]()->void*{const auto choice=in.byte();return choice&8?nullptr:storage.data()+(choice&7);};
  void* a=ptr();void* b=ptr();void* c=ptr();void* bias=ptr();void* work=ptr();
  void* ql=ptr();void* kl=ptr();void* offsets=ptr();
  const int m=in.i32(),n=in.i32(),k=in.i32(),dtype=in.i32(),config=in.i32();
  consume(ti::host::gemm(a,b,bias,c,m,n,k,dtype,config,false));
  consume(ti::host::gemm(a,b,bias,c,m,n,k,dtype,config,true));
  consume(ti::host::workspace(work,m,n,config));consume(ti::host::scales(work));
  consume(ti::host::legacy_gemm(a,b,c,m,n,k));
  const int batch=in.i32(),heads=in.i32(),nq=in.i32(),nk=in.i32(),d=in.i32(),causal=in.i32(),ac=in.i32();
  consume(ti::host::attention(a,b,bias,c,ql,kl,offsets,batch,heads,nq,nk,d,causal,ac));
  consume(ti::host::layernorm(a,b,bias,c,m,n,in.f32()));
  consume(ti::host::softmax(a,c,m,n));consume(ti::host::gelu(a,c,in.i64()));
  return 0;
}
