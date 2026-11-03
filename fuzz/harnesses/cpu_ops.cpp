#include "ik/cpu_kernels.hpp"
#include "input.hpp"
#include <vector>
namespace { volatile float sink=0; }
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data,size_t size) {
  if(size>ti_fuzz::max_input)return 0;
  ti_fuzz::Input in{data,size};const int op=in.byte()%3,m=in.dim(),n=in.dim(),k=in.dim();
  if(op==0) {
    std::vector<float>a(size_t(m)*k),b(size_t(k)*n),c(size_t(m)*n);
    for(auto& x:a)x=in.finite();for(auto& x:b)x=in.finite();
    ik::cpu::matmul_naive(a.data(),b.data(),c.data(),m,n,k);sink=c.back();
    ik::cpu::matmul_blocked(a.data(),b.data(),c.data(),m,n,k);sink=c.back();
  } else {
    std::vector<float>x(size_t(m)*n),y(size_t(m)*n),g(n),b(n);
    for(auto& v:x)v=in.finite();for(auto& v:g)v=in.finite();for(auto& v:b)v=in.finite();
    if(op==1) {
      ik::cpu::softmax_naive(x.data(),y.data(),m,n);sink=y.back();
      ik::cpu::softmax_fused(x.data(),y.data(),m,n);sink=y.back();
    } else {
      const float eps=1e-5f;
      ik::cpu::layernorm_naive(x.data(),g.data(),b.data(),y.data(),m,n,eps);sink=y.back();
      ik::cpu::layernorm_fused(x.data(),g.data(),b.data(),y.data(),m,n,eps);sink=y.back();
    }
  }
  return 0;
}
