#include "tinyinfer_host_codec.h"
#include "tinyinfer_quantization_scalar.h"
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#include <cstdint>

namespace {
static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559,
              "host codec requires IEEE binary32");
bool matrix(size_t elements, size_t columns) {
  return elements>0 && columns>0 && elements%columns==0 &&
         elements<=size_t(std::numeric_limits<int64_t>::max())/sizeof(float);
}
struct Span { uintptr_t begin,end; };
bool span(const void* p,size_t count,size_t width,size_t alignment,Span& out) {
  if(!p || reinterpret_cast<uintptr_t>(p)%alignment || count>SIZE_MAX/width) return false;
  const auto address=reinterpret_cast<uintptr_t>(p);const size_t bytes=count*width;
  if(bytes>UINTPTR_MAX-address) return false;
  out={address,address+bytes};return true;
}
bool distinct(Span a,Span b) { return a.end<=b.begin || b.end<=a.begin; }
float half_float(uint16_t h) {
  const float magnitude=(h&0x7c00u)==0 ? std::ldexp(float(h&0x3ffu),-24) :
    std::ldexp(float(1024u+(h&0x3ffu)),int((h>>10)&31u)-25);
  return (h&0x8000u)?-magnitude:magnitude;
}
uint32_t round_shift(uint32_t value,unsigned shift) {
  if(shift>=32) return 0;
  const uint32_t base=value>>shift;
  const uint32_t mask=(uint32_t(1)<<shift)-1;
  const uint32_t remainder=value&mask, midpoint=uint32_t(1)<<(shift-1);
  return base+uint32_t(remainder>midpoint || (remainder==midpoint && (base&1)));
}
// Input is a finite FP32 value known to round to finite half. No host fenv dependency.
uint16_t float_half(float value) {
  uint32_t bits; std::memcpy(&bits,&value,sizeof bits);
  const uint16_t sign=uint16_t((bits>>16)&0x8000u);
  const int exponent=int((bits>>23)&255u)-127;
  const uint32_t fraction=bits&0x7fffffu;
  if(exponent < -25) return sign;
  if(exponent < -14) return uint16_t(sign|round_shift(fraction|0x800000u,unsigned(-exponent-1)));
  uint32_t rounded=round_shift(fraction,13);
  uint32_t half_exp=uint32_t(exponent+15);
  if(rounded==1024u) { rounded=0; ++half_exp; }
  return uint16_t(sign|(half_exp<<10)|rounded);
}
int round_even(double value) {
  // Finite quotients are bounded near [-127,127]. Python round() uses ties-to-even.
  const double low=std::floor(value), part=value-low;
  int result=int(low);
  if(part>0.5 || (part==0.5 && result%2!=0)) ++result;
  return result;
}
}
extern "C" int ti_host_codec_abi_version(void) { return 1; }
extern "C" int ti_host_quantize_f16(const uint16_t* input,size_t elements,size_t columns,
                                    int8_t* q,size_t q_elements,float* scales,size_t scale_elements) {
  Span in{},out{},scale{};
  if(!matrix(elements,columns) || q_elements!=elements || scale_elements!=columns ||
     std::fegetround()!=FE_TONEAREST || !span(input,elements,2,alignof(uint16_t),in) ||
     !span(q,q_elements,1,alignof(int8_t),out) || !span(scales,scale_elements,4,alignof(float),scale) ||
     !distinct(in,out) || !distinct(in,scale) || !distinct(out,scale))
    return TI_HOST_INVALID;
  // Validate the entire input and all scales before touching caller output.
  for(size_t i=0;i<elements;++i) if(!ti::host::finite_half_bits(input[i])) return TI_HOST_INVALID;
  for(size_t c=0;c<columns;++c) {
    float maximum=0;
    for(size_t i=c;i<elements;i+=columns) maximum=std::fmax(maximum,std::fabs(half_float(input[i])));
    const float scale=maximum?float(double(maximum)/127.0):1.0f;
    if(!std::isfinite(scale) || scale<=0) return TI_HOST_INVALID;
  }
  for(size_t c=0;c<columns;++c) {
    float maximum=0;
    for(size_t i=c;i<elements;i+=columns) maximum=std::fmax(maximum,std::fabs(half_float(input[i])));
    scales[c]=maximum?float(double(maximum)/127.0):1.0f;
  }
  for(size_t i=0;i<elements;++i) {
    // Match Python's binary64 division of exact FP16 and stored FP32 operands.
    const int value=round_even(double(half_float(input[i]))/double(scales[i%columns]));
    q[i]=int8_t(value < -127?-127:(value>127?127:value));
  }
  return TI_HOST_OK;
}
extern "C" int ti_host_dequantize_i8(const int8_t* q,size_t elements,size_t columns,
                                    const float* scales,size_t scale_elements,uint16_t* output,size_t output_elements) {
  Span in{},scale{},out{};
  if(!matrix(elements,columns) || scale_elements!=columns || output_elements!=elements ||
     std::fegetround()!=FE_TONEAREST || !span(q,elements,1,alignof(int8_t),in) ||
     !span(scales,scale_elements,4,alignof(float),scale) || !span(output,output_elements,2,alignof(uint16_t),out) ||
     !distinct(in,scale) || !distinct(in,out) || !distinct(scale,out))
    return TI_HOST_INVALID;
  for(size_t c=0;c<columns;++c) if(!std::isfinite(scales[c]) || scales[c]<=0) return TI_HOST_INVALID;
  for(size_t i=0;i<elements;++i) {
    if(q[i]==std::numeric_limits<int8_t>::min()) return TI_HOST_INVALID;
    const float value=ti::host::dequantize_scalar(q[i],scales[i%columns]);
    // 65520 is the RN-even boundary which becomes infinity in IEEE half.
    if(!std::isfinite(value) || std::fabs(value)>=65520.0f) return TI_HOST_INVALID;
  }
  for(size_t i=0;i<elements;++i) output[i]=float_half(ti::host::dequantize_scalar(q[i],scales[i%columns]));
  return TI_HOST_OK;
}
