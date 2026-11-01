#include "tinyinfer_backend.h"
#include "ik/cpu_kernels.hpp"
#ifdef IK_HAVE_CUDA
#include "ik/cuda_kernels.hpp"
#endif
extern "C" void ti_cpu_gemm(const float* a,const float* b,float* y,int m,int k,int n) { ik::cpu::matmul_blocked(a,b,y,m,n,k); }
extern "C" void ti_cpu_softmax(const float* x,float* y,int r,int c) { ik::cpu::softmax_fused(x,y,r,c); }
extern "C" void ti_cpu_layernorm(const float* x,const float* g,const float* b,float* y,int r,int c,float e) { ik::cpu::layernorm_fused(x,g,b,y,r,c,e); }
#ifdef IK_HAVE_CUDA
extern "C" void ti_cuda_gemm(const float* a,const float* b,float* y,int m,int k,int n) { ik::cuda::matmul_tiled_reg(a,b,y,m,n,k); }
extern "C" void ti_cuda_softmax(const float* x,float* y,int r,int c) { ik::cuda::softmax_fused(x,y,r,c); }
extern "C" void ti_cuda_layernorm(const float* x,const float* g,const float* b,float* y,int r,int c,float e) { ik::cuda::layernorm_fused(x,g,b,y,r,c,e); }
#endif
