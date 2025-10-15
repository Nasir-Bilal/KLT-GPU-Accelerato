#ifndef CUDA_CODE_H
#define CUDA_CODE_H

#ifdef __cplusplus
extern "C" {
#endif

void runMyCudaKernel(float *input, float *output, int size);

#ifdef __cplusplus
}
#endif

#endif
