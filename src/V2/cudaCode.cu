#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
        return; \
    }

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

    int idx = (y - bordery) / step * ((ncols - 2*borderx + step - 1)/step) * 3
            + (x - borderx) / step * 3;

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
    int *pointlist,              // output array [x, y, val, x, y, val, ...]
    const float *gradx,          // gradient X
    const float *grady,          // gradient Y
    int ncols, int nrows,
    int borderx, int bordery,
    int window_hw, int window_hh,
    int nSkippedPixels)
{
    // -------------------------------------------------------------------------
    // 1️⃣ Derived constants
    // -------------------------------------------------------------------------
    unsigned int limit = (1u << (sizeof(int) * 8 - 1)) - 1;
    int step = nSkippedPixels + 1;

    int numX = (ncols - 2 * borderx + step - 1) / step;
    int numY = (nrows - 2 * bordery + step - 1) / step;
    int maxPoints = numX * numY;

    size_t gradSize = (size_t)ncols * nrows * sizeof(float);
    size_t pointlistSize = (size_t)maxPoints * 3 * sizeof(int);

    printf("[GPU] Launching kernel (%d×%d grid, %d×%d threads)\n", numX, numY, 16, 16);
    printf("[GPU] Allocating %.2f MB total\n",
           (gradSize * 2 + pointlistSize) / (1024.0 * 1024.0));

    // -------------------------------------------------------------------------
    // 2️⃣ Allocate device memory
    // -------------------------------------------------------------------------
    float *d_gradx = nullptr, *d_grady = nullptr;
    int *d_pointlist = nullptr;

    CUDA_CHECK(cudaMalloc(&d_gradx, gradSize));
    CUDA_CHECK(cudaMalloc(&d_grady, gradSize));
    CUDA_CHECK(cudaMalloc(&d_pointlist, pointlistSize));

    // -------------------------------------------------------------------------
    // 3️⃣ Copy gradients to device
    // -------------------------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(d_gradx, gradx, gradSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grady, grady, gradSize, cudaMemcpyHostToDevice));

    // -------------------------------------------------------------------------
    // 4️⃣ Define launch configuration
    // -------------------------------------------------------------------------
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (numX + blockSize.x - 1) / blockSize.x,
        (numY + blockSize.y - 1) / blockSize.y
    );

    // -------------------------------------------------------------------------
    // 5️⃣ Launch kernel
    // -------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // 6️⃣ Copy results back to host
    // -------------------------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(pointlist, d_pointlist, pointlistSize, cudaMemcpyDeviceToHost));

    // -------------------------------------------------------------------------
    // 7️⃣ Free device memory
    // -------------------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_gradx));
    CUDA_CHECK(cudaFree(d_grady));
    CUDA_CHECK(cudaFree(d_pointlist));

    printf("[GPU] Done.\n");
}
