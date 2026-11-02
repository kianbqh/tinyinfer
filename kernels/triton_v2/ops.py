"""Independent Triton FP16 GEMM and fused softmax baselines for v2 only."""
from __future__ import annotations
import torch
import triton
import triton.language as tl


@triton.jit
def _gemm(A, B, C, M:tl.constexpr, N:tl.constexpr, K:tl.constexpr,
          BM:tl.constexpr, BN:tl.constexpr, BK:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    acc=tl.full((BM,BN),0,tl.float32)
    for k0 in range(tl.cdiv(K,BK)):
        indices=k0*BK+kk
        a=tl.load(A+rows[:,None]*K+indices[None,:],(rows[:,None]<M)&(indices[None,:]<K),0)
        b=tl.load(B+indices[:,None]*N+cols[None,:],(indices[:,None]<K)&(cols[None,:]<N),0)
        acc=tl.dot(a,b,acc)
    tl.store(C+rows[:,None]*N+cols[None,:],acc,(rows[:,None]<M)&(cols[None,:]<N))


@triton.jit
def _softmax(X,Y,N:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0)
    cols=tl.arange(0,BLOCK)
    x=tl.load(X+row*N+cols,cols<N,-float("inf")).to(tl.float32)
    z=tl.exp(x-tl.max(x,axis=0))
    tl.store(Y+row*N+cols,z/tl.sum(z,axis=0),cols<N)


def _tensor(x,name,ndim=None):
    if not isinstance(x,torch.Tensor) or x.device.type!="cuda" or x.dtype!=torch.float16:
        raise TypeError(f"{name} must be a CUDA FP16 tensor")
    if not x.is_contiguous() or x.numel()==0 or x.requires_grad or (ndim is not None and x.ndim!=ndim):
        raise ValueError(f"{name} must be nonempty contiguous inference-only rank {ndim}")


def gemm(a,b,*,config=0,output_dtype=None):
    _tensor(a,"a",2);_tensor(b,"b",2)
    if a.device!=b.device or a.shape[1]!=b.shape[0]:
        raise ValueError("GEMM device/inner dimensions differ")
    configs={0:(32,32,32,4),1:(64,32,32,4),2:(64,64,32,8)}
    if type(config) is not int or config not in configs:
        raise ValueError("Triton GEMM config must be 0,1,2")
    dtype=torch.float16 if output_dtype is None else output_dtype
    if dtype not in (torch.float16,torch.float32):
        raise TypeError("output_dtype must be FP16 or FP32")
    m,k=a.shape;n=b.shape[1]
    bm,bn,bk,warps=configs[config]
    with torch.cuda.device(a.device):
        stream=torch.cuda.current_stream(a.device)
        out=torch.empty((m,n),dtype=dtype,device=a.device)
        for tensor in (a,b,out):tensor.record_stream(stream)
        _gemm[(triton.cdiv(m,bm),triton.cdiv(n,bn))](a,b,out,m,n,k,bm,bn,bk,num_warps=warps)
    return out


def softmax(x):
    _tensor(x,"x")
    if x.ndim==0 or x.shape[-1]>32768:
        raise ValueError("softmax requires last dimension 1..32768")
    n=x.shape[-1];rows=x.numel()//n;block=triton.next_power_of_2(n)
    with torch.cuda.device(x.device):
        stream=torch.cuda.current_stream(x.device)
        out=torch.empty_like(x)
        for tensor in (x,out):tensor.record_stream(stream)
        _softmax[(rows,)](x,out,n,block,num_warps=8 if block>=2048 else 4)
    return out
