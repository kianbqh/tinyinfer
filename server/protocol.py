"""Version 1, network-order u32 length prefix; no pickle, text or tokenization."""
from __future__ import annotations
import struct

VERSION,GENERATE,CANCEL,FINAL,FIRST=1,1,2,0x8001,0x8002
REQUEST=struct.Struct("!HHQII")
RESPONSE=struct.Struct("!HHQIIIIQQQQ")
MAX_RESPONSE=60+4*1024


def request(identifier,tokens,max_new_tokens):
    if type(identifier) is not int or not 0<identifier<2**64:raise ValueError("request ID")
    if not tokens or any(type(t) is not int or not 0<=t<50257 for t in tokens):raise ValueError("tokens")
    if type(max_new_tokens) is not int or max_new_tokens<1 or len(tokens)+max_new_tokens>1024:raise ValueError("context")
    body=REQUEST.pack(VERSION,GENERATE,identifier,max_new_tokens,len(tokens))+struct.pack(f"!{len(tokens)}I",*tokens)
    return struct.pack("!I",len(body))+body


def cancellation(identifier):
    if type(identifier) is not int or not 0<identifier<2**64:raise ValueError("request ID")
    body=REQUEST.pack(VERSION,CANCEL,identifier,0,0)
    return struct.pack("!I",len(body))+body


def decode_response(body):
    if len(body)<RESPONSE.size:raise ValueError("response size")
    v,kind,identifier,status,count,batch,reserved,queue,first,infer,total=RESPONSE.unpack_from(body)
    if v!=VERSION or kind not in (FIRST,FINAL) or not identifier or status not in range(5) or reserved or len(body)!=RESPONSE.size+4*count:
        raise ValueError("response schema")
    if batch>32 or count>1024 or (kind==FIRST and (status or count!=1)):raise ValueError("response bounds")
    tokens=list(struct.unpack_from(f"!{count}I",body,RESPONSE.size)) if count else []
    if any(t>=50257 for t in tokens):raise ValueError("response token")
    return dict(kind=kind,id=identifier,status=status,tokens=tokens,batch_size=batch,
                queue_ns=queue,first_ns=first,infer_ns=infer,server_ns=total)


async def read_response(reader):
    size=struct.unpack("!I",await reader.readexactly(4))[0]
    if not RESPONSE.size<=size<=MAX_RESPONSE:raise ValueError("response frame limit")
    return decode_response(await reader.readexactly(size))
