#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
        return; \
    }
#include <cstdio>
#include <cassert>
#include "cudaCode.h"

// ---------------------------
// Device function
// ---------------------------
__device__ float minEigenvalue(float gxx, float gxy, float gyy)
{
    return 0.5f * (gxx + gyy - sqrtf((gxx - gyy) * (gxx - gyy) + 4.0f * gxy * gxy));
}

// ---------------------------
// CUDA Kernel
// ---------------------------
__global__ void KLTSelectGoodFeaturesKernel(
    int *pointlist,
    const float *gradx,
    const float *grady,
    int ncols, int nrows,
    int borderx, int bordery,
    int window_hw, int window_hh,
    int nSkippedPixels,
    unsigned int limit)
{
    int step = nSkippedPixels + 1;

    int x = borderx + (blockIdx.x * blockDim.x + threadIdx.x) * step;
    int y = bordery + (blockIdx.y * blockDim.y + threadIdx.y) * step;

    if (x >= ncols - borderx || y >= nrows - bordery)
        return;
    float gxx = 0.0f, gxy = 0.0f, gyy = 0.0f;

    // Sum gradients in window
    for (int yy = y - window_hh; yy <= y + window_hh; yy++)
        for (int xx = x - window_hw; xx <= x + window_hw; xx++) {
            float gx = gradx[yy * ncols + xx];
            float gy = grady[yy * ncols + xx];
            gxx += gx * gx;
            gxy += gx * gy;
            gyy += gy * gy;
        }

    float val = minEigenvalue(gxx, gxy, gyy);
    if (val > limit) val = (float) limit;

     //Calculate number of points in X direction
    int numX = (ncols - 2*borderx + step - 1) / step;
    
    //Linear index of this thread's point
    int linearIdx = ((y - bordery) / step) * numX + ((x - borderx) / step);
    int idx = linearIdx * 3;

    pointlist[idx]     = x;
    pointlist[idx + 1] = y;
    pointlist[idx + 2] = (int) val;
}

// ---------------------------
// Wrapper Function (handles all GPU ops)
// ---------------------------
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
        return; \
    }

extern "C" void launchKLTSelectGoodFeatures(
    int *pointlist,
    const float *gradx,          
    const float *grady,          
    int ncols, int nrows,
    int borderx, int bordery,
    int window_hw, int window_hh,
    int nSkippedPixels)
{

    unsigned int limit = (1u << (sizeof(int) * 8 - 1)) - 1;
    int step = nSkippedPixels + 1;

    int numX = (ncols - 2 * borderx + step - 1) / step;
    int numY = (nrows - 2 * bordery + step - 1) / step;
    int maxPoints = numX * numY;

    int gradSize = (int)ncols * nrows * sizeof(float);
    int pointlistSize = (int)maxPoints * 3 * sizeof(int);

    float *d_gradx = nullptr, *d_grady = nullptr;
    int *d_pointlist = nullptr;

    CUDA_CHECK(cudaMalloc(&d_gradx, gradSize));
    CUDA_CHECK(cudaMalloc(&d_grady, gradSize));
    CUDA_CHECK(cudaMalloc(&d_pointlist, pointlistSize));

    CUDA_CHECK(cudaMemcpy(d_gradx, gradx, gradSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grady, grady, gradSize, cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (numX + blockSize.x - 1) / blockSize.x,
        (numY + blockSize.y - 1) / blockSize.y
    );

    KLTSelectGoodFeaturesKernel<<<gridSize, blockSize>>>(
        d_pointlist,
        d_gradx, d_grady,
        ncols, nrows,
        borderx, bordery,
        window_hw, window_hh,
        nSkippedPixels,
        limit
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(pointlist, d_pointlist, pointlistSize, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_gradx));
    CUDA_CHECK(cudaFree(d_grady));
    CUDA_CHECK(cudaFree(d_pointlist));

    printf("[GPU] Done.\n");
}
    
/*--------------- _convolveImageHoriz ---------------*/
/*
#define CUDA_CHECK(call) \
  do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

typedef struct {
    float *data;  
    int ncols;
    int nrows;
} KLT_FloatImage;

typedef struct {
    float *data;   
    int width;
} ConvolutionKernel;
*/
//KERNEL
__global__ void convolve_horiz_kernel(const float* imgin,
                           const float* kernel_data,
                           float* imgout,
                           int ncols, int nrows,
                           int kernelWidth)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= nrows || col >= ncols) return;

    int radius = kernelWidth / 2;

    // Replicate CPU border handling: zero left/right columns within radius
    if (col < radius || col >= (ncols - radius)) {
        imgout[row * ncols + col] = 0.0f;
        return;
    }

    // ppp = ptrrow + i - radius;
    // sum = 0.0;
    // for (k = kernel.width-1 ; k >= 0 ; k--)
    //   sum += *ppp++ * kernel.data[k];
    float sum = 0.0f;
    int p = col - radius;                    // p corresponds to offset (i - radius)
    int rowBase = row * ncols;
    // iterate k from kernelWidth-1 down to 0 to match original ordering exactly
    for (int k = kernelWidth - 1; k >= 0; --k) {
        float v = imgin[rowBase + p];       // *ppp
        float w = kernel_data[k];           // kernel.data[k]
        sum += v * w;
        ++p;                                // ppp++
    }

    imgout[rowBase + col] = sum;
}

//WRAPPER
void convolve_horiz_cuda(const float* h_imgin, const float* h_kernel, float* h_imgout, int ncols, int nrows, int kernelWidth)
{
    assert(kernelWidth % 2 == 1);                  // same assertion as CPU
    // We'll assume imgin != imgout on the host; this wrapper requires separate buffers.

    size_t imgBytes = (size_t)ncols * nrows * sizeof(float);
    size_t kernelBytes = (size_t)kernelWidth * sizeof(float);

    float *d_imgin = nullptr, *d_imgout = nullptr, *d_kernel = nullptr;
    CUDA_CHECK(cudaMalloc(&d_imgin, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_imgout, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernelBytes));

    CUDA_CHECK(cudaMemcpy(d_imgin, h_imgin, imgBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, kernelBytes, cudaMemcpyHostToDevice));
    // Note: we don't initialize d_imgout (kernel writes every pixel)

    // Choose a simple 2D block size
    dim3 block(16, 16); // simple choice, easy to understand
    dim3 grid( (ncols + block.x - 1) / block.x,
               (nrows + block.y - 1) / block.y );

    convolve_horiz_kernel<<<grid, block>>>(d_imgin, d_kernel, d_imgout, ncols, nrows, kernelWidth);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_imgout, d_imgout, imgBytes, cudaMemcpyDeviceToHost));

    cudaFree(d_imgin);
    cudaFree(d_imgout);
    cudaFree(d_kernel);
}

/*--------------- _convolveImageVert ---------------*/

//KERNEL
__global__ void convolve_vert_kernel(const float* __restrict__ imgin, const float* __restrict__ kernel_data, float* __restrict__ imgout, int ncols, int nrows, int kernelWidth)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= ncols || row >= nrows) return;

    int radius = kernelWidth / 2;

    // Border handling: top/bottom rows within radius are zero
    if (row < radius || row >= (nrows - radius)) {
        imgout[row * ncols + col] = 0.0f;
        return;
    }

    // Compute convolution in column direction.
    // Mirror CPU order: for (k = kernel.width-1; k >= 0; k--) sum += *ppp++ * kernel.data[k];
    float sum = 0.0f;
    int p_row = row - radius;        // starting row index for ppp
    int base_index = col;            // column offset for indexing: idx = p_row * ncols + col
    for (int k = kernelWidth - 1; k >= 0; --k) {
        float v = imgin[p_row * ncols + base_index]; // *ppp
        float w = kernel_data[k];
        sum += v * w;
        ++p_row; // ppp += ncols (move one row down)
    }

    imgout[row * ncols + col] = sum;
}

//WRAPPER
void convolve_vert_cuda(const float* h_imgin, const float* h_kernel, float* h_imgout, int ncols, int nrows, int kernelWidth)
{
    assert(kernelWidth % 2 == 1);

    size_t imgBytes = (size_t)ncols * (size_t)nrows * sizeof(float);
    size_t kernelBytes = (size_t)kernelWidth * sizeof(float);

    float *d_imgin = NULL, *d_imgout = NULL, *d_kernel = NULL;
    CUDA_CHECK(cudaMalloc(&d_imgin, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_imgout, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernelBytes));

    CUDA_CHECK(cudaMemcpy(d_imgin, h_imgin, imgBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, kernelBytes, cudaMemcpyHostToDevice));

    // choose simple 2D block/grid
    dim3 block(16, 16);
    dim3 grid( (ncols + block.x - 1) / block.x,
               (nrows + block.y - 1) / block.y );

    convolve_vert_kernel<<<grid, block>>>(d_imgin, d_kernel, d_imgout, ncols, nrows, kernelWidth);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_imgout, d_imgout, imgBytes, cudaMemcpyDeviceToHost));

    cudaFree(d_imgin);
    cudaFree(d_imgout);
    cudaFree(d_kernel);
}