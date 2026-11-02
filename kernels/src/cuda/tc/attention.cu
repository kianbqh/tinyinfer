#include "launchers.cuh"
#include <math_constants.h>
#include <cstddef>
#include <cmath>
namespace ti::v2 {
namespace {
// Four warps own four queries and share one key/value tile. Online FP32 state
// remains in registers; no QxK allocation. This uses SIMT dots, not tensor cores.
template<int D,int TILE>
__global__ void online_attention(const half* q,const half* k,const half* v,half* out,
                                 const int32_t* q_lengths,const int32_t* kv_lengths,
                                 const int32_t* offsets,int heads,int q_len,int kv_len,bool causal) {
  const int t=threadIdx.x,lane=t%32,warp=t/32;
  const int qi=blockIdx.x*4+warp,h=blockIdx.y,b=blockIdx.z;
  const int q_valid=max(0,min(q_len,q_lengths[b]));
  const int kv_valid=max(0,min(kv_len,kv_lengths[b]));
  const size_t qbase=(size_t(b)*heads+h)*q_len*D+size_t(qi)*D;
  const size_t kvbase=(size_t(b)*heads+h)*kv_len*D;
  const bool active=qi<q_valid;
  // Block-wide loop bound is identical even for padded warps; no divergent barrier.
  const int last_query=min(q_valid-1,int(blockIdx.x)*4+3);
  const int64_t last_key=causal?int64_t(offsets[b])+last_query:int64_t(kv_valid)-1;
  const int64_t bound=last_key+1;
  const int end=last_query<0 || bound<0?0:(bound>kv_valid?kv_valid:int(bound));
  __shared__ half keys[TILE][D],values[TILE][D];
  float query[D/32],acc[D/32]={};
  #pragma unroll
  for(int z=0;z<D/32;z++) query[z]=active?__half2float(q[qbase+lane+32*z]):0.f;
  float m=-CUDART_INF_F,l=0.f;
  for(int64_t k0=0;k0<end;k0+=TILE) {
    for(int i=t;i<TILE*D;i+=128) {
      const int64_t key=k0+i/D;const int d=i%D;
      keys[i/D][d]=key<end?k[kvbase+size_t(key)*D+d]:__float2half(0.f);
      values[i/D][d]=key<end?v[kvbase+size_t(key)*D+d]:__float2half(0.f);
    }
    __syncthreads();
    for(int j=0;j<TILE && k0+j<end;j++) {
      if(active && (!causal || int64_t(k0+j)<=int64_t(offsets[b])+qi)) {
        float score=0.f;
        #pragma unroll
        for(int z=0;z<D/32;z++) score+=query[z]*__half2float(keys[j][lane+32*z]);
        #pragma unroll
        for(int delta=16;delta>0;delta>>=1) score+=__shfl_down_sync(0xffffffff,score,delta);
        score=__shfl_sync(0xffffffff,score,0)*rsqrtf(float(D));
        const float next=fmaxf(m,score),alpha=l==0.f?0.f:expf(m-next),beta=expf(score-next);
        #pragma unroll
        for(int z=0;z<D/32;z++) acc[z]=acc[z]*alpha+beta*__half2float(values[j][lane+32*z]);
        l=l*alpha+beta;m=next;
      }
    }
    __syncthreads();
  }
  if(qi<q_len) {
    #pragma unroll
    for(int z=0;z<D/32;z++) out[qbase+lane+32*z]=__float2half_rn(l>0.f?acc[z]/l:0.f);
  }
}
}
cudaError_t attention(const half* q,const half* k,const half* v,half* out,const int32_t* ql,
                      const int32_t* kl,const int32_t* off,int b,int h,int nq,int nk,int d,
                      bool causal,int config,cudaStream_t stream) {
  dim3 grid((nq-1)/4+1,h,b);
  if(d==64 && config==0) online_attention<64,32><<<grid,128,0,stream>>>(q,k,v,out,ql,kl,off,h,nq,nk,causal);
  else if(d==64 && config==1) online_attention<64,64><<<grid,128,0,stream>>>(q,k,v,out,ql,kl,off,h,nq,nk,causal);
  else if(d==128 && config==0) online_attention<128,32><<<grid,128,0,stream>>>(q,k,v,out,ql,kl,off,h,nq,nk,causal);
  else if(d==128 && config==1) online_attention<128,64><<<grid,128,0,stream>>>(q,k,v,out,ql,kl,off,h,nq,nk,causal);
  else return cudaErrorInvalidValue;
  return cudaGetLastError();
}
}
