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

/*--------------- _convolveImageHoriz ---------------*/
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
    
/*--------------- _convolveImageVert ---------------*/
// Simple mapping wrapper: launch CUDA kernel that computes vertical convolution.
// - h_imgin: pointer to input image pixels (row-major, size = ncols * nrows)
// - h_kernel: pointer to kernel data (length = kernelWidth)
// - h_imgout: pointer to output image buffer (row-major, size = ncols * nrows)
// - ncols, nrows: image dimensions
// - kernelWidth: must be odd
void convolve_vert_cuda(const float* h_imgin,
    const float* h_kernel,
    float* h_imgout,
    int ncols,
    int nrows,
    int kernelWidth);

// Optional CPU reference implementation of vertical convolution (for verification)
void convolve_vert_cpu(const float* imgin,
   const float* kernel,
   float* imgout,
   int ncols,
   int nrows,
   int kernelWidth);

#ifdef __cplusplus
}
#endif

#endif // CUDA_CODE_H