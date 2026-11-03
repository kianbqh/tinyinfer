#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <ostream>
#ifdef TI_SERVER_NATIVE
#include "tinyinfer_llm.h"
#endif
namespace ti::server {
class Model {
  bool native_; int capacity_; std::vector<int32_t> lengths_,last_;
#ifdef TI_SERVER_NATIVE
  ti_llm_model* model_=nullptr;
  void check(int status) { if(status) throw std::runtime_error(ti_llm_last_error()); }
#endif
public:
  Model(bool native,const std::string& manifest,int batch,int capacity,int device,int gemm,int attention):
    native_(native),capacity_(capacity),lengths_(batch),last_(batch) {
    if(!native_) return;
#ifdef TI_SERVER_NATIVE
    if(ti_llm_abi_version()!=1) throw std::runtime_error("ti_llm ABI mismatch");
    ti_llm_options options{sizeof(ti_llm_options),device,batch,batch,capacity,gemm,attention};
    check(ti_llm_create(manifest.c_str(),&options,&model_));
#else
    (void)manifest;(void)device;(void)gemm;(void)attention;
    throw std::runtime_error("native support not built");
#endif
  }
  ~Model() {
#ifdef TI_SERVER_NATIVE
    if(model_) ti_llm_destroy(model_);
#endif
  }
  void report_trace(std::ostream& out) noexcept {
#ifdef TI_SERVER_NATIVE
    if(native_) {
      ti_llm_trace t{};int status=ti_llm_get_trace(model_,&t);
      out<<"{\"native_trace_status\":"<<status;
      if(status==0) out<<",\"forward_calls\":"<<t.forward_calls<<",\"embedding\":"<<t.embedding
        <<",\"gemm_fp16\":"<<t.gemm_fp16<<",\"gemm_int8\":"<<t.gemm_int8<<",\"layernorm\":"<<t.layernorm
        <<",\"gelu\":"<<t.gelu<<",\"attention\":"<<t.attention<<",\"residual\":"<<t.residual
        <<",\"kv_scatter_gather\":"<<t.kv_scatter_gather<<",\"lm_head\":"<<t.lm_head
        <<",\"argmax\":"<<t.argmax<<",\"nll\":"<<t.nll;
      out<<"}\n";return;
    }
#endif
    out<<"{\"trace_mode\":\"cpu_oracle_no_native_calls\"}\n";
  }
  void reset(int slot) {
#ifdef TI_SERVER_NATIVE
    if(native_) { check(ti_llm_reset(model_,slot)); return; }
#endif
    lengths_.at(slot)=0; last_.at(slot)=0;
  }
  std::vector<int32_t> forward(const std::vector<int32_t>& slots,const std::vector<std::vector<int32_t>>& rows) {
    if(slots.size()!=rows.size()||slots.empty()) throw std::runtime_error("batch shape");
    std::vector<int32_t> next(slots.size());
#ifdef TI_SERVER_NATIVE
    if(native_) {
      int stride=0; std::vector<int32_t> lens;
      for(auto& row:rows) { stride=std::max(stride,static_cast<int>(row.size())); lens.push_back(row.size()); }
      std::vector<int32_t> tokens(rows.size()*stride,0);
      for(size_t i=0;i<rows.size();++i) std::copy(rows[i].begin(),rows[i].end(),tokens.begin()+i*stride);
      check(ti_llm_forward_batch(model_,slots.data(),tokens.data(),lens.data(),slots.size(),stride,nullptr,next.data()));
      return next;
    }
#endif
    for(size_t i=0;i<slots.size();++i) {
      int s=slots[i];
      if(rows[i].empty()||lengths_.at(s)+rows[i].size()>static_cast<size_t>(capacity_)) throw std::runtime_error("CPU context limit");
      for(auto t:rows[i]) { last_[s]=t; ++lengths_[s]; }
      next[i]=(last_[s]+lengths_[s]+1)%50257;
    }
    return next;
  }
};
}
