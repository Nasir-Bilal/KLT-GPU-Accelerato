#include <cuda_runtime.h>
#include "cudaCode.h"
#include <stdio.h>

__global__ void calculateGradientSelectGoodFeatures(
    float *d_gradDataX, float *d_gradDataY, int ncols, int nrows, float *gxx, float *gxy, float *gyy
    ) {
    int xx = blockIdx.x * blockDim.x + threadIdx.x;
    int yy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = ncols * yy + xx;
   

    if (xx < ncols && yy < nrows){
        float gx = *(d_gradDataX + idx);
        float gy = *(d_gradDataY + idx);
        atomicAdd(gxx, gx * gx);
        atomicAdd(gxy, gx * gy);
        atomicAdd(gyy, gy * gy);

    }
        
}

void runCalculateGradient(float *gradDataX, float *gradDataY, int ncols, int nrows) {
    float *d_gradDataX, *d_gradDataY;
    float *gxx = 0, *gxy = 0, *gyy = 0; 
    float h_gxx = 0, h_gxy = 0, h_gyy = 0;
    

    cudaMalloc(&d_gradDataX ,sizeof(float) * ncols * nrows );
    cudaMalloc(&d_gradDataY ,sizeof(float) * ncols * nrows);
    cudaMalloc(&d_gxx, sizeof(float));
    cudaMalloc(&d_gxy, sizeof(float));
    cudaMalloc(&d_gyy, sizeof(float));

    cudaMemcpy( d_gradDataX, gradDataX, sizeof(float) * ncols * nrows, cudaMemcpyHostToDevice)
    cudaMemcpy( d_gradDataY, gradDataY, sizeof(float) * ncols * nrows, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gxx, &h_gxx, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gxy, &h_gxy, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gyy, &h_gyy, sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16,16);
    dim3 gridSize (
        (ncols + blockSize.x - 1) / blockSize.x,
        (nrows + blockSize.y - 1)/ blockSize.y 
    );

    //manage the output.
    calculateGradientSelectGoodFeatures<<<gridSize , blockSize>>>(
        d_gradDataX, d_gradDataY,ncols,nrows, gxx, gxy, gyy 
        );

    cudaDeviceSynchronize();  // Wait for GPU to finish


    cudaMemcpy(&h_gxx, d_gxx, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_gxy, d_gxy, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_gyy, d_gyy, sizeof(float), cudaMemcpyDeviceToHost);

   
    cudaFree(d_gradDataX);
    cudaFree(d_gradDataY);
    cudaFree(d_gxx);
    cudaFree(d_gxy);
    cudaFree(d_gyy);
   
}
