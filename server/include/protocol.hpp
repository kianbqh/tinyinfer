#pragma once
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ti::server {
constexpr uint16_t version=1, generate=1, cancel=2, final_response=0x8001, first_response=0x8002;
enum Status:uint32_t { ok=0, invalid=1, overloaded=2, cancelled=3, internal=4 };
using Bytes=std::vector<uint8_t>;
inline uint64_t read_be(const uint8_t* p,unsigned n) {
  uint64_t v=0; for(unsigned i=0;i<n;++i) v=(v<<8)|p[i]; return v;
}
inline void append_be(Bytes& v,uint64_t n,unsigned count) {
  for(unsigned i=count;i>0;--i) v.push_back(static_cast<uint8_t>(n>>(8*(i-1))));
}
struct WireRequest { uint16_t kind; uint64_t id; uint32_t max_new; std::vector<int32_t> tokens; };
inline WireRequest parse_request(const Bytes& b,uint32_t capacity,uint32_t max_new,uint32_t vocab=50257) {
  if(b.size()<20||read_be(b.data(),2)!=version) throw std::runtime_error("request version/size");
  WireRequest r{static_cast<uint16_t>(read_be(b.data()+2,2)),read_be(b.data()+4,8),
                static_cast<uint32_t>(read_be(b.data()+12,4)),{}};
  const uint64_t count=read_be(b.data()+16,4);
  if(!r.id||count>(UINT32_MAX-20)/4||b.size()!=20+count*4) throw std::runtime_error("request ID/length");
  if(r.kind==cancel) { if(count||r.max_new) throw std::runtime_error("cancel fields"); return r; }
  if(r.kind!=generate||!count||!r.max_new||r.max_new>max_new||count+r.max_new>capacity)
    throw std::runtime_error("token/context limits");
  for(size_t i=0;i<count;++i) {
    uint32_t t=static_cast<uint32_t>(read_be(b.data()+20+i*4,4));
    if(t>=vocab) throw std::runtime_error("token ID");
    r.tokens.push_back(static_cast<int32_t>(t));
  }
  return r;
}
struct Response {
  uint16_t kind=final_response; uint64_t id=0; Status status=ok;
  uint32_t batch_size=0; uint64_t queue_ns=0,first_ns=0,infer_ns=0,server_ns=0;
  std::vector<int32_t> tokens;
};
inline Bytes encode_response(const Response& r) {
  Bytes b; append_be(b,60+r.tokens.size()*4,4); append_be(b,version,2); append_be(b,r.kind,2);
  append_be(b,r.id,8); append_be(b,r.status,4); append_be(b,r.tokens.size(),4);
  append_be(b,r.batch_size,4); append_be(b,0,4);
  for(auto n:{r.queue_ns,r.first_ns,r.infer_ns,r.server_ns}) append_be(b,n,8);
  for(auto t:r.tokens) append_be(b,static_cast<uint32_t>(t),4);
  return b;
}
// Incremental decoder bounded even before the complete prefix/body arrives.
class Framer {
  Bytes data_; uint32_t maximum_;
public:
  explicit Framer(uint32_t max):maximum_(max) {}
  template<class Fn> void feed(const uint8_t* p,size_t n,Fn on_frame) {
    for(size_t i=0;i<n;++i) {
      data_.push_back(p[i]);
      if(data_.size()<4) continue;
      uint64_t length=read_be(data_.data(),4);
      if(length<20||length>maximum_) throw std::runtime_error("frame length limit");
      if(data_.size()==length+4) { Bytes body(data_.begin()+4,data_.end()); data_.clear(); on_frame(std::move(body)); }
    }
  }
};
}
