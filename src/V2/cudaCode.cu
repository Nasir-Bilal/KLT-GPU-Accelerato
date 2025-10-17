// src/V2/cudaCode.cu
#include <cuda_runtime.h>
#include <cstdio>
#include "cudaCode.h"

__device__ float minEigenvalue(float gxx, float gxy, float gyy)
{
    return 0.5f * (gxx + gyy - sqrtf((gxx - gyy) * (gxx - gyy) + 4.0f * gxy * gxy));
}


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

    // Sum gradients in surrounding window
    for (int yy = y - window_hh; yy <= y + window_hh; yy++) {
        for (int xx = x - window_hw; xx <= x + window_hw; xx++) {
            float gx = gradx[yy * ncols + xx];
            float gy = grady[yy * ncols + xx];
            gxx += gx * gx;
            gxy += gx * gy;
            gyy += gy * gy;
        }
    }

    float val = minEigenvalue(gxx, gxy, gyy);
    if (val > limit) val = (float) limit;

    // Write to pointlist (3 ints per point)
    int idx = (y - bordery) / step * ((ncols - 2*borderx + step - 1)/step) * 3
            + (x - borderx) / step * 3;
    pointlist[idx]     = x;
    pointlist[idx + 1] = y;
    pointlist[idx + 2] = (int) val;
}


void launchKLTSelectGoodFeatures(
    int *d_pointlist,
    const float *d_gradx,
    const float *d_grady,
    int ncols, int nrows,
    int borderx, int bordery,
    int window_hw, int window_hh,
    int nSkippedPixels)
{
    unsigned int limit = (1u << (sizeof(int) * 8 - 1)) - 1;

    int step = nSkippedPixels + 1;

    dim3 blockSize(16, 16);
    dim3 gridSize(
        ( (ncols - 2*borderx + step - 1) / step + blockSize.x - 1) / blockSize.x,
        ( (nrows - 2*bordery + step - 1) / step + blockSize.y - 1) / blockSize.y
    );

    KLTSelectGoodFeaturesKernel<<<gridSize, blockSize>>>(
        d_pointlist, d_gradx, d_grady,
        ncols, nrows,
        borderx, bordery,
        window_hw, window_hh,
        nSkippedPixels,
        limit
    );

    cudaDeviceSynchronize();
}
