"""Actual GPU gates, called only after the coordinated M2 execution window opens."""
from __future__ import annotations
import gc


def run_correctness(ops,torch,*,rtol=0.02,atol=0.02):
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend,sdpa_kernel
    rows=[]
    only=getattr(ops,"catalog",{}).get("only_config",-1)
    gemm_configs=list(range(12)) if only==-1 else [only]

    def gate(name,fn):
        try:
            detail=fn()
            torch.cuda.synchronize()
            rows.append({"name":name,"status":"passed","detail":detail})
        except Exception as error:
            rows.append({"name":name,"status":"failed","error_type":type(error).__name__,"error":str(error)})

    def close(actual,expected):
        torch.testing.assert_close(actual.float(),expected.float(),rtol=rtol,atol=atol)
        return {"max_abs_error":(actual.float()-expected.float()).abs().max().item(),"rtol":rtol,"atol":atol}

    def gemm_case(m,n,k,config,int8=False):
        a=torch.randn((m,k),device="cuda",dtype=torch.float16)*0.25
        b=torch.randn((k,n),device="cuda",dtype=torch.float16)*0.25
        if int8:b[:,0]=0  # zero channel exercises the no-zero-point dequant path
        bias=torch.randn(n,device="cuda",dtype=torch.float16)*0.1
        if int8:
            scale=b.float().abs().amax(dim=0).div(127).clamp_min(1e-8)
            quant=(b.float()/scale).round().clamp(-127,127).to(torch.int8)
            dequant=(quant.float()*scale).half()
            out=ops.gemm_int8(a,quant,scale,bias=bias,config=config,output_dtype=torch.float32)
            ref=a.float()@dequant.float()+bias.float()
        else:
            out=ops.gemm(a,b,bias=bias,config=config,output_dtype=torch.float32)
            ref=a.float()@b.float()+bias.float()
        detail=close(out,ref)
        # FP16 output uses the same epilogue and has a separate store path.
        half_out=(ops.gemm_int8(a,quant,scale,bias=bias,config=config) if int8 else ops.gemm(a,b,bias=bias,config=config))
        close(half_out,ref)
        return detail

    for config in gemm_configs:
        for m,n,k in ((1,7,3),(17,33,19),(65,67,63),(128,128,128)):
            for int8 in (False,True):
                gate(f"gemm_{m}_{n}_{k}_config{config}_int8{int8}",lambda m=m,n=n,k=k,c=config,i=int8:gemm_case(m,n,k,c,i))

    def attention_case(d,nq,nk,ql,kl,offsets,causal,config):
        q=torch.randn((2,2,nq,d),device="cuda",dtype=torch.float16)*0.5
        k=torch.randn((2,2,nk,d),device="cuda",dtype=torch.float16)*0.5;v=torch.randn_like(k)
        out=ops.attention(q,k,v,q_lengths=ql,kv_lengths=kl,query_offsets=offsets,causal=causal,config=config)
        qi=torch.arange(nq,device=q.device)[None,:,None]
        ki=torch.arange(nk,device=q.device)[None,None,:]
        mask=(qi<torch.tensor(ql,device=q.device)[:,None,None]) & (ki<torch.tensor(kl,device=q.device)[:,None,None])
        if causal:mask=mask & (ki<=torch.tensor(offsets,device=q.device)[:,None,None]+qi)
        with sdpa_kernel(SDPBackend.MATH):
            ref=F.scaled_dot_product_attention(q,k,v,attn_mask=mask[:,None,:,:],dropout_p=0.0,is_causal=False)
        # SDPA's backend handling of all-masked rows is checked explicitly.
        ref=torch.where(mask.any(dim=-1)[:,None,:,None],ref,0)
        detail=close(out,ref)
        padded=~mask.any(dim=-1)[:,None,:,None].expand_as(out)
        if padded.any() and out[padded].count_nonzero().item()!=0:raise AssertionError("all-masked output must be exactly zero")
        return detail

    for d in (64,128):
        for config in (0,1):
            for nq,nk,ql,kl,off,causal in (
                (17,33,[17,11],[33,0],[0,0],True),
                (1,31,[1,1],[31,19],[30,18],True),
                (5,19,[5,0],[13,19],[7,0],True),
                (9,11,[7,9],[11,8],[0,0],False),
            ):
                gate(f"attention_d{d}_q{nq}_k{nk}_c{config}_causal{causal}",
                     lambda d=d,nq=nq,nk=nk,ql=ql,kl=kl,off=off,c=causal,conf=config:attention_case(d,nq,nk,ql,kl,off,c,conf))

    def elementwise():
        details=[]
        for cols in (1,7,257,768,3072):
            x=torch.randn((3,cols),device="cuda",dtype=torch.float16)
            g=torch.randn(cols,device=x.device,dtype=x.dtype);b=torch.randn_like(g)
            details.append(close(ops.layernorm(x,g,b),F.layer_norm(x.float(),(cols,),g.float(),b.float(),1e-5)))
            details.append(close(ops.gelu_new(x),F.gelu(x.float(),approximate="tanh")))
            details.append(close(ops.softmax(x),torch.softmax(x.float(),dim=-1)))
        return details
    gate("layernorm_gelu_new_softmax",elementwise)

    def streams_and_lifetime():
        default=torch.cuda.current_stream();stream=torch.cuda.Stream()
        a=torch.randn((33,19),device="cuda",dtype=torch.float16)
        b=torch.randn((19,67),device="cuda",dtype=torch.float16)
        expected=a.float()@b.float()
        stream.wait_stream(default)
        with torch.cuda.stream(stream):
            # Queue unrelated work before the external kernel to increase the
            # window in which original-stream allocation reuse could race it.
            blocker=torch.ones((256,256),device="cuda",dtype=torch.float16)*0.001
            for _ in range(8):blocker=blocker@blocker
            actual=ops.gemm(a,b,config=gemm_configs[0],output_dtype=torch.float32)
        del a,b;gc.collect()
        # Inputs were allocated on default and consumed on stream. Their storage
        # must remain protected when original-stream allocations are requested.
        for _ in range(32):
            temporary=torch.empty((33,19),device="cuda",dtype=torch.float16);temporary.fill_(123)
            temporary_b=torch.empty((19,67),device="cuda",dtype=torch.float16);temporary_b.fill_(124)
        del temporary,temporary_b,blocker
        default.wait_stream(stream)
        return close(actual,expected)
    gate("non_default_stream_and_input_lifetime",streams_and_lifetime)

    def invalid_inputs():
        a=torch.ones((2,3),device="cuda",dtype=torch.float16);b=torch.ones((3,4),device="cuda",dtype=torch.float16)
        rejected=0
        for fn in (lambda:ops.gemm(a.cpu(),b),lambda:ops.gemm(a.float(),b),lambda:ops.gemm(a.t(),b),
                   lambda:ops.gemm(a,b,config=True),lambda:ops.gemm(a,b,config=12),
                   lambda:ops.gemm(a,b,bias=torch.zeros(3,device=a.device,dtype=a.dtype)),
                   lambda:ops.gemm(a.requires_grad_(),b)):
            try:fn()
            except (ValueError,TypeError):rejected+=1
            else:raise AssertionError("invalid input accepted")
        q=torch.ones((1,1,1,64),device="cuda",dtype=torch.float16)
        for fn in (lambda:ops.attention(q,q,q,q_lengths=[2]),lambda:ops.attention(q,q,q,query_offsets=[-1]),
                   lambda:ops.attention(q,q,q,kv_lengths=[True]),lambda:ops.attention(q,q,q,causal=1)):
            try:fn()
            except (ValueError,TypeError):rejected+=1
            else:raise AssertionError("invalid attention metadata accepted")
        return {"rejected":rejected}
    gate("invalid_tensor_contracts",invalid_inputs)
    return {"status":"passed" if all(r["status"]=="passed" for r in rows) else "failed","cases":rows}
