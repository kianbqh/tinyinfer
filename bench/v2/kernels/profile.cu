// Standalone CUDA12.4-linked Nsight target: no Torch/libtorch/runtime mixing.
// ncu launch filtering must select the actual kernel, not these host setup calls.
#include "tinyinfer_backend_v2.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

static void checked(cudaError_t status) {
  if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Buffer {
  void* pointer=nullptr;
  explicit Buffer(size_t bytes) { checked(cudaMalloc(&pointer,bytes)); }
  ~Buffer() { if(pointer) cudaFree(pointer); }
  Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
  template<class T>T* as() { return static_cast<T*>(pointer); }
};
static int integer(const char* text,int maximum) {
  char* end=nullptr;long result=std::strtol(text,&end,10);
  if(!text[0] || *end || result<=0 || result>maximum) throw std::runtime_error("invalid integer argument");
  return int(result);
}
static void status(int result) {
  if(result) throw std::runtime_error(std::string("v2 status ")+std::to_string(result)+": "+ti_v2_last_error());
}
int main(int argc,char** argv) {
  try {
    if(argc<3 || argc>6) throw std::runtime_error("usage: profile_v2 gemm|int8|attention size [config:1..12] [head_dim] [launches]");
    std::string operation=argv[1];int size=integer(argv[2],8192);
    int config=argc>3?integer(argv[3],12)-1:0,d=argc>4?integer(argv[4],128):64;
    int launches=argc>5?integer(argv[5],100000):20;
    int device;checked(cudaGetDevice(&device));
    cudaStream_t stream;checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    const uintptr_t raw=reinterpret_cast<uintptr_t>(stream);
    if(operation=="gemm" || operation=="int8") {
      const size_t elements=size_t(size)*size;
      std::vector<half> weights(elements),activation(elements);
      std::vector<int8_t> quant(elements);std::vector<float> scales(size,1.f/127.f);
      for(size_t i=0;i<elements;i++) {
        activation[i]=__float2half(float(int(i%29)-14)/31.f);
        quant[i]=int8_t(int(i%127)-63);weights[i]=__float2half(float(quant[i])/127.f);
      }
      Buffer a(elements*2),b(elements*(operation=="int8"?1:2)),s(size_t(size)*4),out(elements*2),workspace(config>=6?elements*4:4);
      checked(cudaMemcpyAsync(a.pointer,activation.data(),elements*2,cudaMemcpyHostToDevice,stream));
      checked(cudaMemcpyAsync(b.pointer,operation=="int8"?static_cast<void*>(quant.data()):static_cast<void*>(weights.data()),elements*(operation=="int8"?1:2),cudaMemcpyHostToDevice,stream));
      checked(cudaMemcpyAsync(s.pointer,scales.data(),size_t(size)*4,cudaMemcpyHostToDevice,stream));
      for(int i=0;i<launches;i++) {
        float* scratch=config>=6?workspace.as<float>():nullptr;
        if(operation=="int8") status(ti_v2_gemm_ex_i8(a.as<uint16_t>(),b.as<int8_t>(),s.as<float>(),nullptr,out.pointer,scratch,size,size,size,0,config,raw,device));
        else status(ti_v2_gemm_ex_f16(a.as<uint16_t>(),b.as<uint16_t>(),nullptr,out.pointer,scratch,size,size,size,0,config,raw,device));
      }
      checked(cudaStreamSynchronize(stream));
    } else if(operation=="attention") {
      if((d!=64 && d!=128) || config>1) throw std::runtime_error("attention D64/128 and config1/2 only");
      const size_t elements=size_t(size)*d;
      std::vector<half> data(elements);
      for(size_t i=0;i<elements;i++) data[i]=__float2half(float(int(i%23)-11)/31.f);
      Buffer q(elements*2),k(elements*2),v(elements*2),out(elements*2),ql(4),kl(4),off(4);
      const int32_t length=size,offset=0;
      for(Buffer* buffer:{&q,&k,&v})checked(cudaMemcpyAsync(buffer->pointer,data.data(),elements*2,cudaMemcpyHostToDevice,stream));
      checked(cudaMemcpyAsync(ql.pointer,&length,4,cudaMemcpyHostToDevice,stream));
      checked(cudaMemcpyAsync(kl.pointer,&length,4,cudaMemcpyHostToDevice,stream));
      checked(cudaMemcpyAsync(off.pointer,&offset,4,cudaMemcpyHostToDevice,stream));
      for(int i=0;i<launches;i++)status(ti_v2_attention_f16(q.as<uint16_t>(),k.as<uint16_t>(),v.as<uint16_t>(),out.as<uint16_t>(),
           ql.as<int32_t>(),kl.as<int32_t>(),off.as<int32_t>(),1,1,size,size,d,1,config,raw,device));
      checked(cudaStreamSynchronize(stream));
    } else throw std::runtime_error("unknown operation");
    checked(cudaStreamDestroy(stream));
    std::printf("{\"status\":\"completed\",\"operation\":\"%s\",\"size\":%d,\"config\":%d,\"head_dim\":%d,\"launches\":%d}\n",operation.c_str(),size,config,d,launches);
    return 0;
  } catch(const std::exception& error) { std::fprintf(stderr,"profile_v2: %s\n",error.what());return 1; }
}
