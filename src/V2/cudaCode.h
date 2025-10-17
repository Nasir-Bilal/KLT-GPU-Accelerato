// src/V2/cudaCode.h
#ifndef CUDA_CODE_H
#define CUDA_CODE_H

#ifdef __cplusplus
extern "C" {
#endif

//#include <cuda_runtime.h>

// ------------------------
// Device kernel and wrapper
// ------------------------

// Launch the KLT feature selection kernel
// d_pointlist : device pointer to output array (3 ints per feature)
// d_gradx, d_grady : device pointers to gradient images
// ncols, nrows : image dimensions
// borderx, bordery : border offsets
// window_hw, window_hh : half-width/half-height of window
// nSkippedPixels : pixels to skip per iteration
void launchKLTSelectGoodFeatures(
    int *d_pointlist,
    const float *d_gradx,
    const float *d_grady,
    int ncols, int nrows,
    int borderx, int bordery,
    int window_hw, int window_hh,
    int nSkippedPixels);

// Optional example kernel you can call from C code
// Keeps compatibility with the example from your skeleton
void runCudaExampleKernel(float *data, int n);

#ifdef __cplusplus
}
#endif

#endif // CUDA_CODE_H
