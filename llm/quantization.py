"""Small stdlib reference for [input, output] symmetric weight-only INT8."""
import math
import struct


def fp32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def fp16(value):
    return struct.unpack('<e', struct.pack('<e', value))[0]


def validate_matrix(rows):
    if not rows or not rows[0]:
        raise ValueError('matrix must be nonempty')
    width = len(rows[0])
    for row in rows:
        if len(row) != width:
            raise ValueError('ragged matrix')
        for x in row:
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
                raise ValueError('weights must be finite real numbers')
    return len(rows), width


def quantize(rows):
    """FP16 reference weights -> INT8 [in,out], FP32 output-channel scales.

    Round to nearest, ties to even (Python round); no -128/zero point. A zero
    channel has scale1 and q0. Scale is rounded to FP32 BEFORE quantization.
    """
    height, width = validate_matrix(rows)
    values = [[fp16(float(x)) for x in row] for row in rows]
    scales = [fp32(max(abs(values[i][j]) for i in range(height)) / 127.0)
              if any(values[i][j] != 0 for i in range(height)) else 1.0
              for j in range(width)]
    if any(s <= 0 or not math.isfinite(s) for s in scales):
        raise ValueError('scale underflow/overflow')
    q = [[max(-127, min(127, round(values[i][j] / scales[j])))
          for j in range(width)] for i in range(height)]
    return q, scales


def dequantize(q, scales, *, to_fp16=True):
    height, width = validate_matrix(q)
    if len(scales) != width or any(isinstance(s, bool) or not isinstance(s, (int, float))
                                  or not math.isfinite(s) or s <= 0 for s in scales):
        raise ValueError('positive finite output-channel scales required')
    for row in q:
        if any(type(x) is not int or not -127 <= x <= 127 for x in row):
            raise ValueError('q must contain symmetric INT8 integers')
    # CUDA multiplies in FP32, then converts to FP16 before WMMA.
    convert = fp16 if to_fp16 else float
    return [[convert(fp32(q[i][j] * fp32(scales[j]))) for j in range(width)]
            for i in range(height)]
