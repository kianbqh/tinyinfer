#include "launchers.cuh"
#include <cmath>
#include <cstddef>
namespace ti::v2 {
namespace {
__device__ float block_sum(float value,float* scratch) {
  scratch[threadIdx.x]=value;__syncthreads();
  for(int stride=128;stride>0;stride>>=1) {
    if(threadIdx.x<stride) scratch[threadIdx.x]+=scratch[threadIdx.x+stride];
    __syncthreads();
  }
  return scratch[0];
}
__global__ void norm(const half* x,const half* g,const half* b,half* y,int cols,float eps) {
  __shared__ float scratch[256];
  const size_t base=size_t(blockIdx.x)*cols;
  float sum=0.f;
  for(int64_t c=threadIdx.x;c<cols;c+=256) sum+=__half2float(x[base+c]);
  const float mean=block_sum(sum,scratch)/cols;
  __syncthreads();
  float squares=0.f;
  for(int64_t c=threadIdx.x;c<cols;c+=256) { float z=__half2float(x[base+c])-mean;squares+=z*z; }
  const float inv=rsqrtf(block_sum(squares,scratch)/cols+eps);
  for(int64_t c=threadIdx.x;c<cols;c+=256)
    y[base+c]=__float2half_rn((__half2float(x[base+c])-mean)*inv*__half2float(g[c])+__half2float(b[c]));
}
__global__ void gelu(const half* x,half* y,int64_t count) {
  for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(blockDim.x)*gridDim.x) {
    float z=__half2float(x[i]);
    y[i]=__float2half_rn(0.5f*z*(1.f+tanhf(0.7978845608028654f*(z+0.044715f*z*z*z))));
  }
}
__global__ void normalized_exp(const half* x,half* y,int cols) {
  __shared__ float scratch[256];
  const size_t base=size_t(blockIdx.x)*cols;
  float top=-CUDART_INF_F;
  for(int64_t c=threadIdx.x;c<cols;c+=256) top=fmaxf(top,__half2float(x[base+c]));
  scratch[threadIdx.x]=top;__syncthreads();
  for(int s=128;s>0;s>>=1) { if(threadIdx.x<s) scratch[threadIdx.x]=fmaxf(scratch[threadIdx.x],scratch[threadIdx.x+s]);__syncthreads(); }
  top=scratch[0];__syncthreads();
  float total=0.f;
  for(int64_t c=threadIdx.x;c<cols;c+=256) total+=expf(__half2float(x[base+c])-top);
  const float denom=block_sum(total,scratch);
  for(int64_t c=threadIdx.x;c<cols;c+=256) y[base+c]=__float2half_rn(expf(__half2float(x[base+c])-top)/denom);
}
}
cudaError_t layernorm(const half* x,const half* g,const half* b,half* y,int r,int c,float e,cudaStream_t s) {
  norm<<<r,256,0,s>>>(x,g,b,y,c,e);return cudaGetLastError();
}
cudaError_t gelu_new(const half* x,half* y,int64_t n,cudaStream_t s) {
  int blocks=int(n<65535LL*256?(n-1)/256+1:65535);
  gelu<<<blocks,256,0,s>>>(x,y,n);return cudaGetLastError();
}
cudaError_t softmax(const half* x,half* y,int r,int c,cudaStream_t s) {
  normalized_exp<<<r,256,0,s>>>(x,y,c);return cudaGetLastError();
}
}
