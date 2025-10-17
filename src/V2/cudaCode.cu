// src/V2/cudaCode.cu
#include <cuda_runtime.h>
#include <cstdio>
#include "cudaCode.h"

// --- GPU kernel (runs on device) ---
__global__ void exampleKernel(float *d_data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = d_data[idx];
        // simple demo computation
        d_data[idx] = x * x + 1.0f;
    }
}

// --- Host wrapper callable from C ---
extern "C" void runCudaExampleKernel(float *data, int n)
{
    float *d_data;
    size_t bytes = sizeof(float) * n;

    // Allocate device memory
    cudaMalloc(&d_data, bytes);

    // Copy data to device
    cudaMemcpy(d_data, data, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize  = (n + blockSize - 1) / blockSize;

    exampleKernel<<<gridSize, blockSize>>>(d_data, n);
    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(data, d_data, bytes, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_data);

    printf("[CUDA] Kernel executed successfully on %d elements\n", n);
}
    