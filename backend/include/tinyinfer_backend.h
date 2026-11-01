#pragma once
extern "C" {
void ti_cpu_gemm(const float*, const float*, float*, int, int, int);
void ti_cpu_softmax(const float*, float*, int, int);
void ti_cpu_layernorm(const float*, const float*, const float*, float*, int, int, float);
void ti_cuda_gemm(const float*, const float*, float*, int, int, int);
void ti_cuda_softmax(const float*, float*, int, int);
void ti_cuda_layernorm(const float*, const float*, const float*, float*, int, int, float);
}
