#ifndef CUDA_CODE_H
#define CUDA_CODE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

// Example wrapper function you can call from C code (existing)
void runCudaExampleKernel(float *data, int n);

// Simple mapping wrapper: launch CUDA kernel that computes horizontal convolution.
// - h_imgin: pointer to input image pixels (row-major, size = ncols * nrows)
// - h_kernel: pointer to kernel data (length = kernelWidth)
// - h_imgout: pointer to output image buffer (row-major, size = ncols * nrows)
// - ncols, nrows: image dimensions
// - kernelWidth: must be odd
void convolve_horiz_cuda(const float* h_imgin,
                         const float* h_kernel,
                         float* h_imgout,
                         int ncols,
                         int nrows,
                         int kernelWidth);

// Optional CPU reference implementation (same semantics as the GPU mapping version).
// Provided so callers can use it for verification in tests.
void convolve_horiz_cpu(const float* imgin,
                        const float* kernel,
                        float* imgout,
                        int ncols,
                        int nrows,
                        int kernelWidth);

#ifdef __cplusplus
}
#endif

#endif // CUDA_CODE_H