"""Required exact native/Python parity gates, run only after a real host build."""
import argparse
import array
import ctypes
import json
from pathlib import Path
import struct
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from llm.export_weights import quantized_bytes
from llm.host_codec import HostCodec
from llm.quantization import dequantize


def check(library):
    codec=HostCodec(library);cases=[]
    # Every finite half bit pattern, plus fixed matrices giving zero/signed-zero,
    # tiny/max magnitudes and exact half-step ratios. No random sampling required.
    all_half=array.array('f',(struct.unpack('<e',struct.pack('<H',i))[0]
                               for i in range(65536) if i&0x7c00 != 0x7c00))
    matrices=[(all_half,32),(array.array('f',[0.,-0.,1.,-1.,127.,-127.,0.5,-0.5,1.5,-1.5]),1),
              (array.array('f',[0.,2**-24,65504.,-0.,-2**-24,-65504.]),3),
              (array.array('f',[127.,127.,0.5,1.5,2.5,-2.5]),2)]
    for values,columns in matrices:
        actual=codec.quantized_bytes(values,columns);expected=quantized_bytes(values,columns)
        if actual!=expected:raise AssertionError('native/reference quantized bytes differ')
        cases.append({'elements':len(values),'columns':columns,'exact_q_and_fp32_scales':True})
    q=[[-127,-3,-1,0,1,3,127],[1,2,3,4,5,6,7]]
    scales=[2**-24,0.5,1.,1.,1.,2.,float(struct.unpack('<f',struct.pack('<f',65504/127))[0])]
    raw_q=bytes((x%256 for row in q for x in row));raw_s=struct.pack('<7f',*scales)
    expected=b''.join(struct.pack('<e',x) for row in dequantize(q,scales) for x in row)
    if codec.dequantized_half_bytes(raw_q,raw_s,7)!=expected:raise AssertionError('dequantization half bytes differ')
    # Failure atomicity and extent/overlap contracts use actual live buffers.
    source=(ctypes.c_uint16*2)(0x3c00,0x7c00);out=(ctypes.c_int8*2)(17,19);scale=(ctypes.c_float*1)(23.)
    before=bytes(out),bytes(scale)
    if codec.lib.ti_host_quantize_f16(source,2,1,out,2,scale,1)!=1:raise AssertionError('nonfinite accepted')
    if before!=(bytes(out),bytes(scale)):raise AssertionError('rejected input changed outputs')
    source=(ctypes.c_uint16*2)(0x3c00,0x4000)
    before=bytes(source),bytes(out),bytes(scale)
    if codec.lib.ti_host_quantize_f16(source,2,1,out,1,scale,1)!=1:raise AssertionError('extent mismatch accepted')
    if before!=(bytes(source),bytes(out),bytes(scale)):raise AssertionError('extent rejection changed buffers')
    if codec.lib.ti_host_quantize_f16(source,2,1,ctypes.cast(source,ctypes.c_void_p),2,scale,1)!=1:
        raise AssertionError('overlap accepted')
    if before!=(bytes(source),bytes(out),bytes(scale)):raise AssertionError('overlap rejection changed buffers')
    for qvalue,svalue in [(-128,1.),(127,1e10),(1,0.),(1,float('nan')),(1,float('inf'))]:
        qq=(ctypes.c_int8*1)(qvalue);ss=(ctypes.c_float*1)(svalue);yy=(ctypes.c_uint16*1)(0x1234)
        if codec.lib.ti_host_dequantize_i8(qq,1,1,ss,1,yy,1)!=1 or yy[0]!=0x1234:
            raise AssertionError('bad dequantization accepted or changed output')
    # A valid prefix must not be written before a later invalid value is found.
    for values,scales2 in [([1,-128],[1.,1.]),([1,127],[1.,1e10]),
                          ([1,1],[1.,0.]),([1,1],[1.,float('nan')]),
                          ([1,1],[1.,float('inf')])]:
        qq=(ctypes.c_int8*2)(*values);ss=(ctypes.c_float*2)(*scales2)
        yy=(ctypes.c_uint16*2)(0x1234,0x5678);sentinel=bytes(yy)
        if codec.lib.ti_host_dequantize_i8(qq,2,2,ss,2,yy,2)!=1 or bytes(yy)!=sentinel:
            raise AssertionError('later invalid value changed an output prefix')
    return {'status':'passed','quantization_cases':cases,'dequantization_exact_half_bytes':True,
            'invalid_extent_overlap_nonfinite_overflow_failure_atomicity':True}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--library',required=True);a=p.parse_args()
    print(json.dumps(check(a.library),indent=2))
