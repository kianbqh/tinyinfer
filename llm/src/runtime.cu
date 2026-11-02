#include "tinyinfer_llm.h"
#include "tinyinfer_backend_v2.h"
#include "artifact.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <chrono>
#include <climits>
#include <memory>
#include <set>
#include <cstdio>

using ti_llm_detail::Artifact;
using ti_llm_detail::Invalid;
namespace {
constexpr int H=768,NH=12,D=64,L=12,V=50257,SCORE_ROWS=16;
thread_local std::string error_text;
struct GpuError:std::runtime_error {using std::runtime_error::runtime_error;};
void ck(cudaError_t code) {if(code!=cudaSuccess)throw GpuError(cudaGetErrorString(code));}
void native(int code) {if(code)throw GpuError(std::string("v2 kernel: ")+ti_v2_last_error());}
struct DeviceScope {
  int old=-1; bool changed=false;
  explicit DeviceScope(int device) {ck(cudaGetDevice(&old));if(old!=device){ck(cudaSetDevice(device));changed=true;}}
  ~DeviceScope(){if(changed)cudaSetDevice(old);}
};
__global__ void embed(const int32_t* ids,const int32_t* lengths,const int32_t* offsets,
                      const half* wte,const half* wpe,half* x,int rows,int q) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=int64_t(rows)*H)return;
  int row=int(i/H),c=int(i%H),b=row/q,t=row%q;
  x[i]=t<lengths[b]?__float2half_rn(__half2float(wte[int64_t(ids[row])*H+c])+__half2float(wpe[(offsets[b]+t)*H+c])):__float2half_rn(0.f);
}
__global__ void add(const half* a,const half* b,half* out,int64_t count) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count)out[i]=__float2half_rn(__half2float(a[i])+__half2float(b[i]));
}
__global__ void split_scatter(const half* qkv,half* qout,half* kc,half* vc,
                             const int32_t* slots,const int32_t* lengths,const int32_t* offsets,
                             int batch,int q,int cap,int layer) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=int64_t(batch)*NH*q*D)return;
  int d=int(i%D),t=int(i/D%q),h=int(i/(int64_t(D)*q)%NH),b=int(i/(int64_t(D)*q*NH));
  bool valid=t<lengths[b];int64_t src=(int64_t(b)*q+t)*3*H+h*D+d;
  qout[i]=valid?qkv[src]:__float2half_rn(0.f);
  if(valid){int64_t dst=((((int64_t(slots[b])*L+layer)*NH+h)*cap+offsets[b]+t)*D+d);kc[dst]=qkv[src+H];vc[dst]=qkv[src+2*H];}
}
__global__ void gather_cache(const half* kc,const half* vc,half* kp,half* vp,
                            const int32_t* slots,const int32_t* kv_lengths,
                            int batch,int k,int cap,int layer) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=int64_t(batch)*NH*k*D)return;
  int d=int(i%D),t=int(i/D%k),h=int(i/(int64_t(D)*k)%NH),b=int(i/(int64_t(D)*k*NH));
  int64_t src=((((int64_t(slots[b])*L+layer)*NH+h)*cap+t)*D+d);
  bool valid=t<kv_lengths[b];kp[i]=valid?kc[src]:__float2half_rn(0.f);vp[i]=valid?vc[src]:__float2half_rn(0.f);
}
__global__ void merge_heads(const half* in,half* out,int batch,int q) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=int64_t(batch)*q*H)return;
  int c=int(i%H),t=int(i/H%q),b=int(i/(int64_t(H)*q));
  out[i]=in[((int64_t(b)*NH+c/D)*q+t)*D+c%D];
}
__global__ void last_rows(const half* x,half* out,const int32_t* lengths,int batch,int q) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<batch*H){int b=i/H;out[i]=x[(int64_t(b)*q+lengths[b]-1)*H+i%H];}
}
__global__ void argmax(const float* logits,int32_t* out,int rows) {
  __shared__ float values[256];__shared__ int indices[256];int row=blockIdx.x,t=threadIdx.x;
  float best=-INFINITY;int idx=INT_MAX;
  for(int j=t;j<V;j+=256){float v=logits[int64_t(row)*V+j];if(v>best||(v==best&&j<idx)){best=v;idx=j;}}
  values[t]=best;indices[t]=idx;__syncthreads();
  for(int step=128;step;step/=2){if(t<step){float v=values[t+step];int j=indices[t+step];if(v>values[t]||(v==values[t]&&j<indices[t])){values[t]=v;indices[t]=j;}}__syncthreads();}
  if(t==0&&row<rows)out[row]=indices[0];
}
__global__ void nll_rows(const float* logits,const int32_t* tokens,const uint8_t* mask,
                         double* loss,int first_position,int rows) {
  __shared__ float maxima[256];__shared__ double sums[256];int row=blockIdx.x,t=threadIdx.x,pos=first_position+row;
  if(row>=rows)return;
  float m=-INFINITY;for(int j=t;j<V;j+=256)m=fmaxf(m,logits[int64_t(row)*V+j]);maxima[t]=m;__syncthreads();
  for(int step=128;step;step/=2){if(t<step)maxima[t]=fmaxf(maxima[t],maxima[t+step]);__syncthreads();}
  double s=0;for(int j=t;j<V;j+=256)s+=exp(double(logits[int64_t(row)*V+j])-maxima[0]);sums[t]=s;__syncthreads();
  for(int step=128;step;step/=2){if(t<step)sums[t]+=sums[t+step];__syncthreads();}
  if(t==0)loss[pos]=mask[pos]?double(maxima[0])+log(sums[0])-double(logits[int64_t(row)*V+tokens[pos]]):0.;
}
__global__ void sum_losses(const double* loss,double* out,int length) {
  __shared__ double sums[256];int t=threadIdx.x;double value=0;for(int i=t;i<length;i+=256)value+=loss[i];sums[t]=value;__syncthreads();
  for(int step=128;step;step/=2){if(t<step)sums[t]+=sums[t+step];__syncthreads();}if(t==0)*out=sums[0];
}
unsigned blocks(int64_t n){return unsigned((n+255)/256);}
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
} // namespace

struct ti_llm_model {
  ti_llm_options options{};cudaStream_t stream=nullptr;bool poisoned=false;
  std::vector<void*> allocations;std::map<std::string,void*> weights;
  std::vector<int32_t> lengths,host_tokens,host_slots,host_lengths,host_offsets,host_kv;
  ti_llm_trace trace{};bool quantized=false;
  half *x=nullptr,*res=nullptr,*norm=nullptr,*proj=nullptr,*qkv=nullptr,*mlp=nullptr,*act=nullptr;
  half *q=nullptr,*kp=nullptr,*vp=nullptr,*attn=nullptr,*kc=nullptr,*vc=nullptr,*last=nullptr;
  float *logits=nullptr,*workspace=nullptr;int32_t *ids=nullptr,*slots=nullptr,*qlens=nullptr,*offsets=nullptr,*kvlens=nullptr,*next=nullptr,*score_tokens=nullptr;
  uint8_t* score_mask=nullptr;double *loss=nullptr,*loss_sum=nullptr;
  template<class T>T* alloc(size_t count){void* p=nullptr;ck(cudaMalloc(&p,count*sizeof(T)));allocations.push_back(p);return static_cast<T*>(p);}
  ~ti_llm_model(){int previous=-1;cudaGetDevice(&previous);cudaSetDevice(options.device);if(stream)cudaStreamSynchronize(stream);for(void* p:allocations)cudaFree(p);if(stream)cudaStreamDestroy(stream);if(previous>=0)cudaSetDevice(previous);}
  void init(const Artifact& artifact) {
    quantized=artifact.precision=="int8";std::string linear=quantized?"int8":"float16";
    std::set<std::string> required;
    auto tensor=[&](const std::string& name,std::vector<int> shape,const std::string& dtype){
      const auto& info=artifact.require(name,shape,dtype);required.insert(name);
      auto* target=alloc<uint8_t>(info.nbytes);ck(cudaMemcpyAsync(target,artifact.binary.data()+info.offset,info.nbytes,cudaMemcpyHostToDevice,stream));weights[name]=target;
    };
    tensor("transformer.wte.weight",{V,H},"float16");tensor("transformer.wpe.weight",{1024,H},"float16");
    tensor("transformer.ln_f.weight",{H},"float16");tensor("transformer.ln_f.bias",{H},"float16");
    for(int l=0;l<L;++l){std::string p="transformer.h."+std::to_string(l)+".";
      for(auto n:{"ln_1","ln_2"}){tensor(p+n+".weight",{H},"float16");tensor(p+n+".bias",{H},"float16");}
      for(auto pair:std::vector<std::pair<std::string,std::pair<int,int>>>{{"attn.c_attn",{H,3*H}},{"attn.c_proj",{H,H}},{"mlp.c_fc",{H,4*H}},{"mlp.c_proj",{4*H,H}}}){
        tensor(p+pair.first+".weight",{pair.second.first,pair.second.second},linear);tensor(p+pair.first+".bias",{pair.second.second},"float16");
        if(quantized)tensor(p+pair.first+".weight.scales",{pair.second.second},"float32");
      }
    }
    tensor("lm_head.weight",{H,V},linear);if(quantized)tensor("lm_head.weight.scales",{V},"float32");
    if(required.size()!=artifact.tensors.size())throw Invalid("unexpected tensor(s) in artifact");
    size_t r=size_t(options.max_batch)*options.capacity;
    x=alloc<half>(r*H);res=alloc<half>(r*H);norm=alloc<half>(r*H);proj=alloc<half>(r*H);
    qkv=alloc<half>(r*3*H);mlp=alloc<half>(r*4*H);act=alloc<half>(r*4*H);
    q=alloc<half>(r*H);kp=alloc<half>(r*H);vp=alloc<half>(r*H);attn=alloc<half>(r*H);
    size_t cache=size_t(options.cache_slots+1)*L*NH*options.capacity*D;kc=alloc<half>(cache);vc=alloc<half>(cache);
    last=alloc<half>(size_t(options.max_batch)*H);logits=alloc<float>(size_t(std::max(options.max_batch,SCORE_ROWS))*V);
    if(options.gemm_config>=6)workspace=alloc<float>(std::max(r*4*H,size_t(std::max(options.max_batch,SCORE_ROWS))*V));
    ids=alloc<int32_t>(r);slots=alloc<int32_t>(options.max_batch);qlens=alloc<int32_t>(options.max_batch);offsets=alloc<int32_t>(options.max_batch);kvlens=alloc<int32_t>(options.max_batch);next=alloc<int32_t>(options.max_batch);
    score_tokens=alloc<int32_t>(options.capacity);score_mask=alloc<uint8_t>(options.capacity);loss=alloc<double>(options.capacity);loss_sum=alloc<double>(1);
    lengths.assign(options.cache_slots+1,0);ck(cudaStreamSynchronize(stream));
  }
  const half* hp(const std::string& n){return static_cast<const half*>(weights.at(n));}
  void check(){if(poisoned)throw GpuError("model poisoned by earlier execution failure; destroy/recreate required");}
  uintptr_t handle()const{return reinterpret_cast<uintptr_t>(stream);}
  void gemm(const half* a,const std::string& name,void* out,int m,int n,int k,int dtype,bool bias=true){
    const auto* b=bias?reinterpret_cast<const uint16_t*>(hp(name+".bias")):nullptr;
    if(quantized){++trace.gemm_int8;native(ti_v2_gemm_ex_i8(reinterpret_cast<const uint16_t*>(a),static_cast<const int8_t*>(weights.at(name+".weight")),static_cast<const float*>(weights.at(name+".weight.scales")),b,out,workspace,m,n,k,dtype,options.gemm_config,handle(),options.device));}
    else{++trace.gemm_fp16;native(ti_v2_gemm_ex_f16(reinterpret_cast<const uint16_t*>(a),reinterpret_cast<const uint16_t*>(hp(name+".weight")),b,out,workspace,m,n,k,dtype,options.gemm_config,handle(),options.device));}
  }
  void ln(const half* in,half* out,const std::string& name,int rows){++trace.layernorm;native(ti_v2_layernorm_f16(reinterpret_cast<const uint16_t*>(in),reinterpret_cast<const uint16_t*>(hp(name+".weight")),reinterpret_cast<const uint16_t*>(hp(name+".bias")),reinterpret_cast<uint16_t*>(out),rows,H,1e-5f,handle(),options.device));}
  void transform(const std::vector<int32_t>& s,const std::vector<int32_t>& token_rows,const std::vector<int32_t>& lens,int b,int qlen) {
    host_slots=s;host_tokens=token_rows;host_lengths=lens;host_offsets.resize(b);host_kv.resize(b);int k=0;
    for(int i=0;i<b;++i){host_offsets[i]=lengths[s[i]];host_kv[i]=host_offsets[i]+lens[i];k=std::max(k,host_kv[i]);}
    auto copy=[&](void* out,const std::vector<int32_t>& in){ck(cudaMemcpyAsync(out,in.data(),in.size()*sizeof(int32_t),cudaMemcpyHostToDevice,stream));};
    copy(ids,host_tokens);copy(slots,host_slots);copy(qlens,host_lengths);copy(offsets,host_offsets);copy(kvlens,host_kv);
    int rows=b*qlen;++trace.forward_calls;++trace.embedding;
    embed<<<blocks(int64_t(rows)*H),256,0,stream>>>(ids,qlens,offsets,hp("transformer.wte.weight"),hp("transformer.wpe.weight"),x,rows,qlen);ck(cudaGetLastError());
    for(int l=0;l<L;++l){std::string p="transformer.h."+std::to_string(l)+".";
      ln(x,norm,p+"ln_1",rows);gemm(norm,p+"attn.c_attn",qkv,rows,3*H,H,TI_V2_FP16);
      ++trace.kv_scatter_gather;
      split_scatter<<<blocks(int64_t(b)*NH*qlen*D),256,0,stream>>>(qkv,q,kc,vc,slots,qlens,offsets,b,qlen,options.capacity,l);ck(cudaGetLastError());
      gather_cache<<<blocks(int64_t(b)*NH*k*D),256,0,stream>>>(kc,vc,kp,vp,slots,kvlens,b,k,options.capacity,l);ck(cudaGetLastError());
      ++trace.attention;native(ti_v2_attention_f16(reinterpret_cast<const uint16_t*>(q),reinterpret_cast<const uint16_t*>(kp),reinterpret_cast<const uint16_t*>(vp),reinterpret_cast<uint16_t*>(attn),qlens,kvlens,offsets,b,NH,qlen,k,D,1,options.attention_config,handle(),options.device));
      merge_heads<<<blocks(int64_t(rows)*H),256,0,stream>>>(attn,norm,b,qlen);ck(cudaGetLastError());gemm(norm,p+"attn.c_proj",proj,rows,H,H,TI_V2_FP16);
      ++trace.residual;add<<<blocks(int64_t(rows)*H),256,0,stream>>>(x,proj,res,int64_t(rows)*H);ck(cudaGetLastError());
      ln(res,norm,p+"ln_2",rows);gemm(norm,p+"mlp.c_fc",mlp,rows,4*H,H,TI_V2_FP16);
      ++trace.gelu;native(ti_v2_gelu_new_f16(reinterpret_cast<const uint16_t*>(mlp),reinterpret_cast<uint16_t*>(act),int64_t(rows)*4*H,handle(),options.device));
      gemm(act,p+"mlp.c_proj",proj,rows,H,4*H,TI_V2_FP16);
      ++trace.residual;add<<<blocks(int64_t(rows)*H),256,0,stream>>>(res,proj,x,int64_t(rows)*H);ck(cudaGetLastError());
    }
    ln(x,norm,"transformer.ln_f",rows);
  }
  void forward(const int32_t* s,const int32_t* tokens,const int32_t* lens,int b,int stride,float* out,int32_t* greedy,bool internal=false){
    check();if(!s||!tokens||!lens||b<1||b>options.max_batch||stride<1||stride>options.capacity)throw Invalid("invalid batch/host pointers/stride");
    std::set<int32_t> unique;int qlen=0;std::vector<int32_t> ss(s,s+b),ll(lens,lens+b);
    for(int i=0;i<b;++i){if(s[i]<0||s[i]>=options.cache_slots+(internal?1:0)||!unique.insert(s[i]).second||lens[i]<1||lens[i]>stride||lengths[s[i]]+lens[i]>options.capacity)throw Invalid("duplicate/invalid slot, length or capacity");qlen=std::max(qlen,lens[i]);for(int j=0;j<lens[i];++j)if(tokens[int64_t(i)*stride+j]<0||tokens[int64_t(i)*stride+j]>=V)throw Invalid("token ID outside GPT-2 vocabulary");}
    std::vector<int32_t> dense(size_t(b)*qlen,0);for(int i=0;i<b;++i)std::copy(tokens+int64_t(i)*stride,tokens+int64_t(i)*stride+lens[i],dense.begin()+int64_t(i)*qlen);
    transform(ss,dense,ll,b,qlen);
    if(out||greedy){last_rows<<<blocks(int64_t(b)*H),256,0,stream>>>(norm,last,qlens,b,qlen);ck(cudaGetLastError());++trace.lm_head;gemm(last,"lm_head",logits,b,V,H,TI_V2_FP32,false);
      if(out)ck(cudaMemcpyAsync(out,logits,size_t(b)*V*sizeof(float),cudaMemcpyDeviceToHost,stream));
      if(greedy){++trace.argmax;argmax<<<b,256,0,stream>>>(logits,next,b);ck(cudaGetLastError());ck(cudaMemcpyAsync(greedy,next,b*sizeof(int32_t),cudaMemcpyDeviceToHost,stream));}
    }
    ck(cudaStreamSynchronize(stream));for(int i=0;i<b;++i)lengths[s[i]]+=lens[i];
  }
};

namespace {
template<class F>int api(ti_llm_model* model,F&& fn){error_text.clear();try{if(!model)throw Invalid("null model");DeviceScope device(model->options.device);model->check();fn();return TI_LLM_OK;}
 catch(const Invalid& e){error_text=e.what();return TI_LLM_INVALID;}catch(const ti_llm_detail::IOError& e){error_text=e.what();return TI_LLM_IO;}
 catch(const GpuError& e){if(model)model->poisoned=true;error_text=e.what();return TI_LLM_CUDA;}catch(const std::exception& e){if(model)model->poisoned=true;error_text=e.what();return TI_LLM_INTERNAL;}catch(...){if(model)model->poisoned=true;error_text="unknown native failure";return TI_LLM_INTERNAL;}}
}
extern "C" int ti_llm_abi_version(void){return 1;}
extern "C" const char* ti_llm_last_error(void){return error_text.c_str();}
extern "C" int ti_llm_create(const char* path,const ti_llm_options* options,ti_llm_model** out){
  error_text.clear();if(out)*out=nullptr;
  try{if(!path||!options||!out||options->struct_size!=sizeof(ti_llm_options)||options->device<0||options->max_batch<1||options->max_batch>32||options->cache_slots<options->max_batch||options->cache_slots>1024||options->capacity<2||options->capacity>1024||options->gemm_config<0||options->gemm_config>11||options->attention_config<0||options->attention_config>1)throw Invalid("invalid options/ABI size");
    if(ti_v2_abi_version()!=2)throw Invalid("v2 kernel ABI required");auto catalog=ti_llm_detail::Parser(std::string(ti_v2_gemm_catalog_json())).run();int only=int(catalog.at("only_config").real());if(only>=0&&only!=options->gemm_config)throw Invalid("GEMM config not compiled in this backend");
    Artifact artifact(path);DeviceScope device(options->device);auto model=std::make_unique<ti_llm_model>();model->options=*options;ck(cudaStreamCreateWithFlags(&model->stream,cudaStreamNonBlocking));model->init(artifact);*out=model.release();return TI_LLM_OK;
  }catch(const Invalid& e){error_text=e.what();return TI_LLM_INVALID;}catch(const ti_llm_detail::IOError& e){error_text=e.what();return TI_LLM_IO;}catch(const GpuError& e){error_text=e.what();return TI_LLM_CUDA;}catch(const std::exception& e){error_text=e.what();return TI_LLM_INTERNAL;}catch(...){error_text="unknown create failure";return TI_LLM_INTERNAL;}
}
extern "C" int ti_llm_destroy(ti_llm_model* model){error_text.clear();if(!model){error_text="null model";return TI_LLM_INVALID;}delete model;return TI_LLM_OK;}
extern "C" int ti_llm_reset(ti_llm_model* m,int32_t slot){return api(m,[&]{if(slot<0||slot>=m->options.cache_slots)throw Invalid("invalid public slot");m->lengths[slot]=0;});}
extern "C" int ti_llm_lengths(ti_llm_model* m,int32_t* out,int32_t count){return api(m,[&]{if(!out||count!=m->options.cache_slots)throw Invalid("wrong lengths output extent");std::copy(m->lengths.begin(),m->lengths.begin()+count,out);});}
extern "C" int ti_llm_get_trace(ti_llm_model* m,ti_llm_trace* out){return api(m,[&]{if(!out)throw Invalid("null trace output");*out=m->trace;});}
extern "C" int ti_llm_forward_batch(ti_llm_model* m,const int32_t* slots,const int32_t* tokens,const int32_t* lengths,int32_t batch,int32_t stride,float* out,int32_t* next){return api(m,[&]{m->forward(slots,tokens,lengths,batch,stride,out,next);});}
extern "C" int ti_llm_greedy_batch(ti_llm_model* m,const int32_t* slots,const int32_t* prompts,const int32_t* prompt_lengths,const int32_t* requested,int32_t batch,int32_t stride,int32_t* output,int32_t out_stride,int32_t* output_lengths,double* ttft,double* total){
 return api(m,[&]{if(!slots||!prompts||!prompt_lengths||!requested||!output||batch<1||batch>m->options.max_batch||stride<1||stride>m->options.capacity||out_stride<1||out_stride>m->options.capacity)throw Invalid("invalid greedy arguments");
   std::set<int> seen;for(int i=0;i<batch;++i){if(slots[i]<0||slots[i]>=m->options.cache_slots||!seen.insert(slots[i]).second||m->lengths[slots[i]]||prompt_lengths[i]<1||prompt_lengths[i]>stride||requested[i]<1||requested[i]>out_stride||requested[i]+prompt_lengths[i]>m->options.capacity)throw Invalid("greedy needs empty distinct slots and bounded positive lengths");for(int j=0;j<prompt_lengths[i];++j)if(prompts[int64_t(i)*stride+j]<0||prompts[int64_t(i)*stride+j]>=V)throw Invalid("invalid prompt token");}
   std::fill(output,output+int64_t(batch)*out_stride,-1);if(output_lengths)std::fill(output_lengths,output_lengths+batch,0);
   auto start=Clock::now();std::vector<int32_t> next(batch);m->forward(slots,prompts,prompt_lengths,batch,stride,nullptr,next.data());double first=ms(start);
   std::vector<int> done(batch,1);for(int i=0;i<batch;++i){output[int64_t(i)*out_stride]=next[i];if(ttft)ttft[i]=first;if(total&&requested[i]==1)total[i]=first;}
   for(;;){std::vector<int32_t> active,ids,ones;std::vector<int> owner;
     for(int i=0;i<batch;++i)if(done[i]<requested[i]){active.push_back(slots[i]);ids.push_back(next[i]);ones.push_back(1);owner.push_back(i);}if(active.empty())break;
     std::vector<int32_t> generated(active.size());m->forward(active.data(),ids.data(),ones.data(),int(active.size()),1,nullptr,generated.data());double elapsed=ms(start);
     for(size_t j=0;j<owner.size();++j){int i=owner[j];next[i]=generated[j];output[int64_t(i)*out_stride+done[i]]=next[i];++done[i];if(total&&done[i]==requested[i])total[i]=elapsed;}
   }
   if(output_lengths)for(int i=0;i<batch;++i)output_lengths[i]=done[i];
 });
}
extern "C" int ti_llm_score_window(ti_llm_model* m,const int32_t* tokens,const uint8_t* mask,int32_t length,double* nll,uint64_t* count,double* positions){
 return api(m,[&]{if(!tokens||!mask||!nll||!count||length<2||length>m->options.capacity||mask[0])throw Invalid("invalid score window");uint64_t targets=0;for(int i=0;i<length;++i){if(tokens[i]<0||tokens[i]>=V||mask[i]>1)throw Invalid("invalid score token/mask");targets+=mask[i];}if(!targets)throw Invalid("no shifted targets selected");
   int slot=m->options.cache_slots;m->lengths[slot]=0;std::vector<int32_t> ss{slot},ll{length-1},dense(tokens,tokens+length-1);m->transform(ss,dense,ll,1,length-1);
   ck(cudaMemcpyAsync(m->score_tokens,tokens,length*sizeof(int32_t),cudaMemcpyHostToDevice,m->stream));ck(cudaMemcpyAsync(m->score_mask,mask,length,cudaMemcpyHostToDevice,m->stream));ck(cudaMemsetAsync(m->loss,0,length*sizeof(double),m->stream));
   for(int row=0;row<length-1;row+=SCORE_ROWS){int rows=std::min(SCORE_ROWS,length-1-row);++m->trace.lm_head;m->gemm(m->norm+int64_t(row)*H,"lm_head",m->logits,rows,V,H,TI_V2_FP32,false);++m->trace.nll;nll_rows<<<rows,256,0,m->stream>>>(m->logits,m->score_tokens,m->score_mask,m->loss,row+1,rows);ck(cudaGetLastError());}
   sum_losses<<<1,256,0,m->stream>>>(m->loss,m->loss_sum,length);ck(cudaGetLastError());ck(cudaMemcpyAsync(nll,m->loss_sum,sizeof(double),cudaMemcpyDeviceToHost,m->stream));if(positions)ck(cudaMemcpyAsync(positions,m->loss,length*sizeof(double),cudaMemcpyDeviceToHost,m->stream));ck(cudaStreamSynchronize(m->stream));if(!std::isfinite(*nll)||*nll<0)throw GpuError("nonfinite/negative NLL");*count=targets;
 });
}
