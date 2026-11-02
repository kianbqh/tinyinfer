"""Explicit native CPU codec for the actual exporter; no implicit build/fallback."""
import array
import ctypes
from pathlib import Path
import struct
import sys


class HostCodec:
    def __init__(self, library):
        self.path = Path(library).resolve(strict=True)
        self.lib = ctypes.CDLL(str(self.path))
        self.lib.ti_host_codec_abi_version.argtypes = []
        self.lib.ti_host_codec_abi_version.restype = ctypes.c_int
        if self.lib.ti_host_codec_abi_version() != 1:
            raise ValueError('native host codec ABI v1 required')
        p, z = ctypes.c_void_p, ctypes.c_size_t
        self.lib.ti_host_quantize_f16.argtypes = [p,z,z,p,z,p,z]
        self.lib.ti_host_quantize_f16.restype = ctypes.c_int
        self.lib.ti_host_dequantize_i8.argtypes = [p,z,z,p,z,p,z]
        self.lib.ti_host_dequantize_i8.restype = ctypes.c_int
        if sys.byteorder != 'little':
            raise ValueError('exporter codec requires little endian host')

    def quantized_bytes(self, values, columns):
        if type(columns) is not int or columns <= 0 or not values or len(values) % columns:
            raise ValueError('nonempty rectangular [K,N] FP16 reference required')
        # Same IEEE-half packing as the existing exporter; not a quantizer replica.
        raw = bytearray()
        for start in range(0, len(values), 4096):
            chunk = values[start:start+4096]
            raw.extend(struct.pack('<'+'e'*len(chunk), *chunk))
        count = len(values)
        source = (ctypes.c_uint16*count).from_buffer(raw)
        q = (ctypes.c_int8*count)()
        scales = (ctypes.c_float*columns)()
        status = self.lib.ti_host_quantize_f16(source,count,columns,q,count,scales,columns)
        if status:
            raise ValueError(f'native quantization rejected buffers/values ({status})')
        return bytes(q), bytes(scales)

    def dequantized_half_bytes(self, q_bytes, scale_bytes, columns):
        if type(columns) is not int or columns <= 0 or len(scale_bytes) != columns*4:
            raise ValueError('FP32 output-channel scales required')
        count = len(q_bytes)
        if not count or count % columns:
            raise ValueError('nonempty rectangular INT8 matrix required')
        q = (ctypes.c_int8*count).from_buffer_copy(q_bytes)
        scales = (ctypes.c_float*columns).from_buffer_copy(scale_bytes)
        out = (ctypes.c_uint16*count)()
        status = self.lib.ti_host_dequantize_i8(q,count,columns,scales,columns,out,count)
        if status:
            raise ValueError(f'native dequantization rejected buffers/values ({status})')
        return bytes(out)
