#include "tinyinfer_host_codec.h"
#include "input.hpp"
#include <array>
#include <cstdlib>
#include <limits>
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data,size_t size) {
  if(size>ti_fuzz::max_input)return 0;
  ti_fuzz::Input in{data,size};const size_t rows=in.dim(),cols=in.dim(),count=rows*cols;
  std::array<uint16_t,1024> half{},out{};std::array<int8_t,1024> q{};std::array<float,32> scales{};
  for(size_t i=0;i<count;++i)half[i]=in.u16();
  const int status=ti_host_quantize_f16(half.data(),count,cols,q.data(),count,scales.data(),cols);
  if(status==TI_HOST_OK) {
    if(ti_host_dequantize_i8(q.data(),count,cols,scales.data(),cols,out.data(),count)!=TI_HOST_OK)std::abort();
    for(size_t i=0;i<count;++i)if(q[i]==std::numeric_limits<int8_t>::min())std::abort();
  } else if(status!=TI_HOST_INVALID)std::abort();
  // Malformed values use real, correctly-sized buffers; never fabricated pointers.
  for(size_t i=0;i<count;++i)q[i]=int8_t(int(in.byte())-128);
  for(size_t c=0;c<cols;++c)scales[c]=in.f32();
  const int ds=ti_host_dequantize_i8(q.data(),count,cols,scales.data(),cols,out.data(),count);
  if(ds!=TI_HOST_OK && ds!=TI_HOST_INVALID)std::abort();
  // Extent rejection must occur before reads/writes. Buffers remain bounded.
  if(ti_host_quantize_f16(half.data(),count,cols,q.data(),count-1,scales.data(),cols)!=TI_HOST_INVALID)std::abort();
  if(ti_host_dequantize_i8(q.data(),count,cols,scales.data(),cols,out.data(),count-1)!=TI_HOST_INVALID)std::abort();
  return 0;
}
