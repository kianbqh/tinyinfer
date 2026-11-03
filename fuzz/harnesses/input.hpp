#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
namespace ti_fuzz {
constexpr size_t max_input=65536, max_weights=65536;
constexpr int max_cpu_dim=32;
struct Input {
  const uint8_t* data;size_t size,pos=0;
  uint8_t byte(){return pos<size?data[pos++]:0;}
  uint16_t u16(){uint16_t x=byte();return uint16_t(x|(uint16_t(byte())<<8));}
  uint32_t u32(){uint32_t x=0;for(int i=0;i<4;++i)x|=uint32_t(byte())<<(8*i);return x;}
  int32_t i32(){uint32_t bits=u32();int32_t x;std::memcpy(&x,&bits,4);return x;}
  int64_t i64(){uint64_t bits=u32();bits|=uint64_t(u32())<<32;int64_t x;std::memcpy(&x,&bits,8);return x;}
  float f32(){uint32_t bits=u32();float x;std::memcpy(&x,&bits,4);return x;}
  float finite(){return (int(byte())-128)/16.0f;}
  int dim(){return 1+byte()%max_cpu_dim;}
};
}
