#pragma once
// Private, strict ASCII manifest reader and SHA256; no third-party JSON/crypto dependency.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ti_llm_detail {
struct Invalid : std::runtime_error { using std::runtime_error::runtime_error; };
struct IOError : std::runtime_error { using std::runtime_error::runtime_error; };
struct Json {
  enum Kind { Null, Bool, Number, String, Array, Object } kind=Null;
  std::string value;
  std::vector<Json> array;
  std::map<std::string,Json> object;
  const Json& at(const std::string& key) const {
    if(kind!=Object || !object.count(key)) throw Invalid("missing manifest field: "+key);
    return object.at(key);
  }
  std::string str() const { if(kind!=String) throw Invalid("expected JSON string"); return value; }
  uint64_t integer() const {
    if(kind!=Number || value.empty() || value.find_first_not_of("0123456789")!=std::string::npos)
      throw Invalid("expected unsigned exact JSON integer");
    uint64_t out=0;
    for(char c:value) {
      if(out>(std::numeric_limits<uint64_t>::max()-uint64_t(c-'0'))/10) throw Invalid("integer overflow");
      out=out*10+uint64_t(c-'0');
    }
    return out;
  }
  double real() const {
    if(kind!=Number) throw Invalid("expected JSON number");
    size_t used=0; double out=std::stod(value,&used);
    if(used!=value.size() || !std::isfinite(out)) throw Invalid("invalid JSON number");
    return out;
  }
};
class Parser {
  const std::string& text; size_t pos=0;
  void ws() { while(pos<text.size() && (text[pos]==' '||text[pos]=='\t'||text[pos]=='\r'||text[pos]=='\n')) ++pos; }
  char take() { if(pos>=text.size()) throw Invalid("truncated JSON"); return text[pos++]; }
  void expect(char c) { if(take()!=c) throw Invalid("invalid JSON punctuation"); }
  std::string string() {
    expect('"'); std::string out;
    for(;;) {
      char c=take(); if(c=='"') return out;
      if(c=='\\') {
        c=take();
        if(c=='"'||c=='\\'||c=='/') out+=c;
        else if(c=='b') out+='\b'; else if(c=='f') out+='\f'; else if(c=='n') out+='\n';
        else if(c=='r') out+='\r'; else if(c=='t') out+='\t';
        else throw Invalid("manifest strings must use ASCII (unicode escapes unsupported)");
      } else {
        if(static_cast<unsigned char>(c)<32 || static_cast<unsigned char>(c)>126) throw Invalid("non-ASCII manifest string");
        out+=c;
      }
    }
  }
  Json parse(int depth) {
    if(depth>32) throw Invalid("JSON nesting limit"); ws(); if(pos>=text.size()) throw Invalid("empty JSON");
    Json out; char c=text[pos];
    if(c=='{') {
      ++pos; out.kind=Json::Object; ws(); if(pos<text.size()&&text[pos]=='}') {++pos; return out;}
      for(;;) {
        ws(); auto key=string(); ws(); expect(':'); auto item=parse(depth+1);
        if(!out.object.emplace(key,std::move(item)).second) throw Invalid("duplicate JSON key");
        ws(); char delimiter=take(); if(delimiter=='}') break; if(delimiter!=',') throw Invalid("object delimiter");
      }
    } else if(c=='[') {
      ++pos; out.kind=Json::Array; ws(); if(pos<text.size()&&text[pos]==']') {++pos; return out;}
      for(;;) { out.array.push_back(parse(depth+1)); ws(); char delimiter=take(); if(delimiter==']') break; if(delimiter!=',') throw Invalid("array delimiter"); }
    } else if(c=='"') { out.kind=Json::String; out.value=string(); }
    else if(c=='t'||c=='f'||c=='n') {
      std::string literal=c=='t'?"true":c=='f'?"false":"null";
      if(text.compare(pos,literal.size(),literal)!=0) throw Invalid("invalid JSON literal");
      pos+=literal.size(); out.kind=c=='n'?Json::Null:Json::Bool; out.value=literal;
    } else {
      out.kind=Json::Number; size_t begin=pos;
      if(text[pos]=='-') ++pos;
      if(pos>=text.size()) throw Invalid("number expected");
      if(text[pos]=='0') ++pos;
      else { if(text[pos]<'1'||text[pos]>'9') throw Invalid("number expected"); while(pos<text.size()&&text[pos]>='0'&&text[pos]<='9') ++pos; }
      if(pos<text.size()&&text[pos]=='.') { ++pos; size_t start=pos; while(pos<text.size()&&text[pos]>='0'&&text[pos]<='9') ++pos; if(pos==start) throw Invalid("invalid fraction"); }
      if(pos<text.size()&&(text[pos]=='e'||text[pos]=='E')) { ++pos; if(pos<text.size()&&(text[pos]=='+'||text[pos]=='-')) ++pos; size_t start=pos; while(pos<text.size()&&text[pos]>='0'&&text[pos]<='9') ++pos; if(pos==start) throw Invalid("invalid exponent"); }
      out.value=text.substr(begin,pos-begin);
    }
    return out;
  }
public:
  explicit Parser(const std::string& s):text(s) {}
  Json run() { auto out=parse(0); ws(); if(pos!=text.size()) throw Invalid("trailing JSON"); return out; }
};

class Sha256 {
  uint32_t h[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
  uint8_t pending[64]{}; size_t used=0; uint64_t bytes=0;
  static uint32_t rotr(uint32_t x,int n) { return (x>>n)|(x<<(32-n)); }
  void block(const uint8_t* p) {
    static const uint32_t k[64]={0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
      0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,
      0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,
      0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,
      0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
      0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,
      0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    uint32_t w[64];
    for(int i=0;i<16;++i) w[i]=(uint32_t(p[i*4])<<24)|(uint32_t(p[i*4+1])<<16)|(uint32_t(p[i*4+2])<<8)|p[i*4+3];
    for(int i=16;i<64;++i) { auto a=w[i-15],b=w[i-2]; w[i]=w[i-16]+(rotr(a,7)^rotr(a,18)^(a>>3))+w[i-7]+(rotr(b,17)^rotr(b,19)^(b>>10)); }
    uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],z=h[7];
    for(int i=0;i<64;++i) {
      uint32_t t1=z+(rotr(e,6)^rotr(e,11)^rotr(e,25))+((e&f)^(~e&g))+k[i]+w[i];
      uint32_t t2=(rotr(a,2)^rotr(a,13)^rotr(a,22))+((a&b)^(a&c)^(b&c));
      z=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=z;
  }
public:
  void update(const void* data,size_t length) {
    const auto* p=static_cast<const uint8_t*>(data); bytes+=length;
    while(length) { size_t n=std::min(length,64-used); std::memcpy(pending+used,p,n);used+=n;p+=n;length-=n;if(used==64){block(pending);used=0;} }
  }
  std::string finish() {
    uint64_t bits=bytes*8; uint8_t one=0x80,zero=0; update(&one,1);
    while(used!=56) update(&zero,1);
    uint8_t tail[8];for(int i=0;i<8;++i) tail[7-i]=uint8_t(bits>>(i*8));update(tail,8);
    std::ostringstream out;out<<std::hex<<std::setfill('0');for(auto v:h)out<<std::setw(8)<<v;return out.str();
  }
  static std::string digest(const void* data,size_t length) { Sha256 hash;hash.update(data,length);return hash.finish(); }
};
inline std::vector<uint8_t> read_bytes(const std::string& path,uint64_t limit) {
  std::ifstream stream(path,std::ios::binary|std::ios::ate);
  if(!stream) throw IOError("cannot open artifact: "+path);
  auto end=stream.tellg();if(end<0||uint64_t(end)>limit)throw Invalid("artifact exceeds size limit");
  std::vector<uint8_t> data(static_cast<size_t>(end));stream.seekg(0);
  if(!data.empty()&&!stream.read(reinterpret_cast<char*>(data.data()),std::streamsize(data.size())))throw IOError("short artifact read");
  return data;
}
struct TensorInfo { std::string dtype;std::vector<int> shape;uint64_t offset=0,nbytes=0; };
struct Artifact {
  std::vector<uint8_t> binary;
  std::map<std::string,TensorInfo> tensors;
  std::string precision;
  struct Input { Json manifest; std::vector<uint8_t> binary; };
  static Json parse_manifest(const std::string& text) {
    if(text.size()>1024*1024) throw Invalid("artifact exceeds size limit");
    Json manifest=Parser(text).run();
    if(manifest.at("schema_version").integer()!=1||manifest.at("alignment").integer()!=256||manifest.at("binary").str()!="weights.bin"||
       manifest.at("model_id").str()!="openai-community/gpt2"||manifest.at("revision").str()!="607a30d783dfa663caf39e06633721c8d4cfcd7e"||
       manifest.at("source_sha256").str()!="248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707")throw Invalid("unrecognized artifact identity");
    auto& a=manifest.at("architecture");
    if(a.at("layers").integer()!=12||a.at("heads").integer()!=12||a.at("hidden").integer()!=768||a.at("vocab").integer()!=50257||
       a.at("capacity").integer()!=1024||a.at("gelu").str()!="gelu_new"||a.at("layer_norm_eps").real()!=1e-5)throw Invalid("only pinned GPT-2 124M supported");
    auto precision=manifest.at("precision").str();if(precision!="fp16"&&precision!="int8")throw Invalid("unknown precision");
    return manifest;
  }
  static Input file_input(const std::string& path) {
    auto raw=read_bytes(path,1024*1024);
    auto manifest=parse_manifest(std::string(raw.begin(),raw.end()));
    auto slash=path.find_last_of("/\\");std::string parent=slash==std::string::npos?"":path.substr(0,slash+1);
    return {std::move(manifest),read_bytes(parent+"weights.bin",1024ull*1024*1024)};
  }
  explicit Artifact(const std::string& path):Artifact(file_input(path)) {}
  // Same production parser and checks, with caller-owned in-memory input.
  static Artifact from_memory(const std::string& text,std::vector<uint8_t> weights) {
    if(weights.size()>1024ull*1024*1024) throw Invalid("artifact exceeds size limit");
    return Artifact(Input{parse_manifest(text),std::move(weights)});
  }
 private:
  explicit Artifact(Input input):binary(std::move(input.binary)) {
    const Json& manifest=input.manifest;precision=manifest.at("precision").str();
    if(binary.size()!=manifest.at("binary_bytes").integer()||Sha256::digest(binary.data(),binary.size())!=manifest.at("binary_sha256").str())throw Invalid("weights binary SHA256/length mismatch");
    auto& entries=manifest.at("tensors");if(entries.kind!=Json::Array||entries.array.size()>256)throw Invalid("invalid tensor list");
    std::vector<std::pair<uint64_t,uint64_t>> ranges;
    for(auto& entry:entries.array) {
      TensorInfo t;t.dtype=entry.at("dtype").str();t.offset=entry.at("offset").integer();t.nbytes=entry.at("nbytes").integer();
      uint64_t count=1;auto& shape=entry.at("shape");if(shape.kind!=Json::Array||shape.array.empty()||shape.array.size()>2)throw Invalid("tensor rank");
      for(auto& dim:shape.array){auto n=dim.integer();if(!n||n>50257)throw Invalid("tensor dimension");count*=n;t.shape.push_back(int(n));}
      int bytes=t.dtype=="float16"?2:t.dtype=="float32"?4:t.dtype=="int8"?1:0;
      if(!bytes||t.nbytes!=count*bytes||t.offset%256||t.offset>binary.size()||t.nbytes>binary.size()-t.offset)throw Invalid("tensor extent/dtype/alignment");
      if(Sha256::digest(binary.data()+t.offset,t.nbytes)!=entry.at("sha256").str())throw Invalid("tensor SHA256 mismatch");
      if(!tensors.emplace(entry.at("name").str(),t).second)throw Invalid("duplicate tensor name");
      ranges.emplace_back(t.offset,t.offset+t.nbytes);
      if(t.dtype=="float16") {
        for(uint64_t i=0;i<t.nbytes;i+=2){uint16_t v=uint16_t(binary[t.offset+i])|(uint16_t(binary[t.offset+i+1])<<8);if((v&0x7c00)==0x7c00)throw Invalid("nonfinite half weight");}
      } else if(t.dtype=="float32") {
        for(uint64_t i=0;i<t.nbytes;i+=4){float v;std::memcpy(&v,binary.data()+t.offset+i,4);if(!std::isfinite(v)||v<=0)throw Invalid("nonpositive/nonfinite scale");}
      } else for(uint64_t i=0;i<t.nbytes;++i)if(binary[t.offset+i]==128)throw Invalid("unsupported INT8 -128 weight");
    }
    std::sort(ranges.begin(),ranges.end());for(size_t i=1;i<ranges.size();++i)if(ranges[i-1].second>ranges[i].first)throw Invalid("overlapping weights");
  }
 public:
  const TensorInfo& require(const std::string& name,const std::vector<int>& shape,const std::string& dtype) const {
    auto it=tensors.find(name);if(it==tensors.end()||it->second.shape!=shape||it->second.dtype!=dtype)throw Invalid("tensor schema mismatch: "+name);return it->second;
  }
};
} // namespace ti_llm_detail
