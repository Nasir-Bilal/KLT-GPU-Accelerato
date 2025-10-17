// src/V2/cudaCode.h
#ifndef CUDA_CODE_H
#define CUDA_CODE_H

#ifdef __cplusplus
extern "C" {
#endif

// Example wrapper function you can call from C code
void runCudaExampleKernel(float *data, int n);

#ifdef __cplusplus
}
#endif

#endif
