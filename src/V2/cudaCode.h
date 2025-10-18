#ifndef CUDA_CODE_H
#define CUDA_CODE_H

#include "klt_util.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef float *_FloatWindow;

#include <stddef.h>

// Example wrapper function you can call from C code (existing)
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

/*--------------- _gpu_computeIntensityDifference ---------------*/

void gpu_computeIntensityDifference(_KLT_FloatImage img1,
    _KLT_FloatImage img2,
    float x1, float y1,
    float x2, float y2,
    int width, int height,
    _FloatWindow imgdiff);

/*--------------- _gpu_computeGradientSum ---------------*/
void gpu_computeGradientSum(_KLT_FloatImage gradx1,
_KLT_FloatImage grady1,
_KLT_FloatImage gradx2,
_KLT_FloatImage grady2,
float x1, float y1,
float x2, float y2,
int width, int height,
_FloatWindow gradx,
_FloatWindow grady);

/*--------------- _gpu_computeIntensityDifferenceLightingInsensitive ---------------*/
void gpu_computeIntensityDifferenceLightingInsensitive(_KLT_FloatImage img1,
                       _KLT_FloatImage img2,
                       float x1, float y1,
                       float x2, float y2,
                       int width, int height,
                       _FloatWindow imgdiff);

/*--------------- _gpu_computeGradientSumLightingInsensitive ---------------*/
void gpu_computeGradientSumLightingInsensitive(_KLT_FloatImage gradx1,
               _KLT_FloatImage grady1,
               _KLT_FloatImage gradx2,
               _KLT_FloatImage grady2,
               _KLT_FloatImage img1,
               _KLT_FloatImage img2,
               float x1, float y1,
               float x2, float y2,
               int width, int height,
               _FloatWindow gradx,
               _FloatWindow grady);

/*--------------- _gpu_am_getGradientWinAffine ---------------*/
void gpu_am_getGradientWinAffine(_KLT_FloatImage in_gradx,
 _KLT_FloatImage in_grady,
 float x, float y,
 float Axx, float Ayx, float Axy, float Ayy,
 int width, int height,
 _FloatWindow out_gradx,
 _FloatWindow out_grady);

/*--------------- _gpu_am_computeAffineMappedImage ---------------*/
void gpu_am_computeAffineMappedImage(_KLT_FloatImage img,
     float x, float y,
     float Axx, float Ayx, float Axy, float Ayy,
     int width, int height,
     _FloatWindow imgdiff);

/*--------------- _gpu_am_computeIntensityDifferenceAffine ---------------*/
void gpu_am_computeIntensityDifferenceAffine(_KLT_FloatImage img1,
             _KLT_FloatImage img2,
             float x1, float y1,
             float x2, float y2,
             float Axx, float Ayx, float Axy, float Ayy,
             int width, int height,
             _FloatWindow imgdiff);


#ifdef __cplusplus
}
#endif

#endif // CUDA_CODE_H