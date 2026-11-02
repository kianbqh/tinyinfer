#include "launchers.cuh"
#include "catalog.cuh"
#include "tinyinfer_quantization_scalar.h"
#include <mma.h>
#include <cstddef>
namespace ti::v2 {
namespace {
namespace wmma = nvcuda::wmma;

template<int BM,int BN,int BK,int WARPS,bool INT8>
__device__ void load_stage(half* as,half* bs,const half* a,const void* b,const float* scales,
                          int m,int n,int k,int row,int col,int64_t k0) {
  for(int i=threadIdx.x;i<BM*BK;i+=WARPS*32) {
    int64_t r=int64_t(row)+i/BK,c=k0+i%BK;
    as[i]=(r<m && c<k)?a[size_t(r)*k+c]:__float2half(0.f);
  }
  for(int i=threadIdx.x;i<BK*BN;i+=WARPS*32) {
    int64_t r=k0+i/BN,c=int64_t(col)+i%BN;
    float value=0.f;
    if(r<k && c<n) {
      if constexpr(INT8) value=ti::host::dequantize_scalar(static_cast<const int8_t*>(b)[size_t(r)*n+c],scales[c]);
      else value=__half2float(static_cast<const half*>(b)[size_t(r)*n+c]);
    }
    bs[i]=__float2half_rn(value);
  }
}

template<int BM,int BN,int BK,int WARPS,bool INT8,int STAGES>
__global__ void wmma_gemm(const half* a,const void* b,const float* scales,const half* bias,
                          void* output,int m,int n,int k,int output_dtype) {
  static_assert(BM%16==0 && BN%16==0 && BK%16==0);
  constexpr int TILES=(BM/16)*(BN/16), PER_WARP=(TILES+WARPS-1)/WARPS;
  __shared__ __align__(32) half as[STAGES][BM*BK];
  __shared__ __align__(32) half bs[STAGES][BK*BN];
  __shared__ __align__(32) float cs[BM*BN];
  wmma::fragment<wmma::accumulator,16,16,16,float> acc[PER_WARP];
  const int warp=threadIdx.x/32,row=blockIdx.y*BM,col=blockIdx.x*BN;
  #pragma unroll
  for(int t=0;t<PER_WARP;t++) wmma::fill_fragment(acc[t],0.f);
  if constexpr(STAGES==2) load_stage<BM,BN,BK,WARPS,INT8>(as[0],bs[0],a,b,scales,m,n,k,row,col,0);
  __syncthreads();
  int stage=0;
  for(int64_t k0=0;k0<k;k0+=BK) {
    // Synchronous alternating staging, not cp.async: visibility/reuse is barrier protected.
    if constexpr(STAGES==1) load_stage<BM,BN,BK,WARPS,INT8>(as[0],bs[0],a,b,scales,m,n,k,row,col,k0);
    else if(k0+BK<k) load_stage<BM,BN,BK,WARPS,INT8>(as[stage^1],bs[stage^1],a,b,scales,m,n,k,row,col,k0+BK);
    __syncthreads();
    #pragma unroll
    for(int kk=0;kk<BK;kk+=16) {
      #pragma unroll
      for(int t=0;t<PER_WARP;t++) {
        const int tile=warp+t*WARPS;
        if(tile<TILES) {
          const int tr=(tile/(BN/16))*16,tc=(tile%(BN/16))*16;
          wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> af;
          wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bf;
          wmma::load_matrix_sync(af,as[stage]+tr*BK+kk,BK);
          wmma::load_matrix_sync(bf,bs[stage]+kk*BN+tc,BN);
          wmma::mma_sync(acc[t],af,bf,acc[t]);
        }
      }
    }
    __syncthreads();
    if constexpr(STAGES==2) stage^=1;
  }
  #pragma unroll
  for(int t=0;t<PER_WARP;t++) {
    const int tile=warp+t*WARPS;
    if(tile<TILES) wmma::store_matrix_sync(cs+(tile/(BN/16))*16*BN+(tile%(BN/16))*16,
                                         acc[t],BN,wmma::mem_row_major);
  }
  __syncthreads();
  for(int i=threadIdx.x;i<BM*BN;i+=WARPS*32) {
    int64_t r=int64_t(row)+i/BN,c=int64_t(col)+i%BN;
    if(r<m && c<n) {
      float value=cs[i]+(bias?__half2float(bias[c]):0.f);
      if(output_dtype==1) static_cast<float*>(output)[size_t(r)*n+c]=value;
      else static_cast<half*>(output)[size_t(r)*n+c]=__float2half_rn(value);
    }
  }
}

// Same numerical algorithm as the frozen 64x64 FP32 register-tiled launcher;
// separate v2 symbol merely adds stream ownership and non-aborting launch errors.
__global__ void legacy_f32(const float* a,const float* b,float* c,int m,int n,int k) {
  __shared__ float as[16][64],bs[16][64];
  int tx=threadIdx.x,ty=threadIdx.y,tid=ty*16+tx;
  int row=blockIdx.y*64,col=blockIdx.x*64;
  float acc[4][4]={};
  for(int64_t k0=0;k0<k;k0+=16) {
    #pragma unroll
    for(int l=0;l<4;l++) {
      int i=tid+l*256;int64_t ar=int64_t(row)+i/16,ak=k0+i%16,bk=k0+i/64,bc=int64_t(col)+i%64;
      as[i%16][i/16]=(ar<m && ak<k)?a[size_t(ar)*k+ak]:0.f;
      bs[i/64][i%64]=(bk<k && bc<n)?b[size_t(bk)*n+bc]:0.f;
    }
    __syncthreads();
    #pragma unroll
    for(int kk=0;kk<16;kk++) {
      float av[4],bv[4];
      #pragma unroll
      for(int i=0;i<4;i++) av[i]=as[kk][ty*4+i];
      #pragma unroll
      for(int j=0;j<4;j++) bv[j]=bs[kk][tx*4+j];
      #pragma unroll
      for(int i=0;i<4;i++)
        #pragma unroll
        for(int j=0;j<4;j++) acc[i][j]+=av[i]*bv[j];
    }
    __syncthreads();
  }
  #pragma unroll
  for(int i=0;i<4;i++)
    #pragma unroll
    for(int j=0;j<4;j++) if(row+ty*4+i<m && col+tx*4+j<n)
      c[size_t(row+ty*4+i)*n+col+tx*4+j]=acc[i][j];
}
__global__ void bias_epilogue(const float* x,const half* bias,void* y,int64_t count,int n,int dtype) {
  for(int64_t i=int64_t(blockIdx.x)*256+threadIdx.x;i<count;i+=int64_t(gridDim.x)*256) {
    float value=x[i]+(bias?__half2float(bias[i%n]):0.f);
    if(dtype==1) static_cast<float*>(y)[i]=value;
    else static_cast<half*>(y)[i]=__float2half_rn(value);
  }
}
template<bool INT8,int CONFIG>
void launch_config(const half* a,const void* b,const float* scales,const half* bias,void* out,
                   int m,int n,int k,int dtype,cudaStream_t stream) {
  constexpr int tile=config_tile(CONFIG),stages=config_stages(CONFIG);
  if constexpr(tile==0) wmma_gemm<32,32,16,4,INT8,stages><<<dim3((n-1)/32+1,(m-1)/32+1),128,0,stream>>>(a,b,scales,bias,out,m,n,k,dtype);
  if constexpr(tile==1) wmma_gemm<64,32,32,8,INT8,stages><<<dim3((n-1)/32+1,(m-1)/64+1),256,0,stream>>>(a,b,scales,bias,out,m,n,k,dtype);
  if constexpr(tile==2) wmma_gemm<64,64,32,8,INT8,stages><<<dim3((n-1)/64+1,(m-1)/64+1),256,0,stream>>>(a,b,scales,bias,out,m,n,k,dtype);
}
template<bool INT8>
cudaError_t dispatch(const half* a,const void* b,const float* scales,const half* bias,void* out,
                     int m,int n,int k,int dtype,int config,cudaStream_t stream) {
  // if constexpr excludes every other instantiation in a single-candidate build.
  #define TI_CASE(ID) case ID: if constexpr(TI_V2_ONLY_GEMM_CONFIG==-1 || TI_V2_ONLY_GEMM_CONFIG==ID) { launch_config<INT8,ID>(a,b,scales,bias,out,m,n,k,dtype,stream); } else return cudaErrorInvalidValue; break
  switch(config) {
    TI_CASE(0);TI_CASE(1);TI_CASE(2);TI_CASE(3);TI_CASE(4);TI_CASE(5);
    TI_CASE(6);TI_CASE(7);TI_CASE(8);TI_CASE(9);TI_CASE(10);TI_CASE(11);
    default:return cudaErrorInvalidValue;
  }
  #undef TI_CASE
  return cudaGetLastError();
}
}
cudaError_t gemm(const half* a,const void* b,const float* s,const half* bias,void* c,float* workspace,int m,int n,int k,
                 int dtype,int config,bool int8,cudaStream_t stream) {
  const bool separate=config_separate(config);
  void* target=separate?static_cast<void*>(workspace):c;
  const half* fused_bias=separate?nullptr:bias;
  const int intermediate_dtype=separate?1:dtype;
  cudaError_t status=int8?dispatch<true>(a,b,s,fused_bias,target,m,n,k,intermediate_dtype,config,stream):
                          dispatch<false>(a,b,s,fused_bias,target,m,n,k,intermediate_dtype,config,stream);
  if(status!=cudaSuccess || !separate) return status;
  int64_t count=int64_t(m)*n;int blocks=int(count<65535LL*256?(count-1)/256+1:65535);
  bias_epilogue<<<blocks,256,0,stream>>>(workspace,bias,c,count,n,dtype);
  return cudaGetLastError();
}
cudaError_t legacy_gemm(const float* a,const float* b,float* c,int m,int n,int k,cudaStream_t s) {
  legacy_f32<<<dim3((n-1)/64+1,(m-1)/64+1),dim3(16,16),0,s>>>(a,b,c,m,n,k);
  return cudaGetLastError();
}
}
