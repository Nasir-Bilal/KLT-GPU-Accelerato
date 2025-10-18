// src/V2/cudaCode.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include "cudaCode.h"

#define CUDA_CHECK(call) \
  do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)


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
  void runCudaExampleKernel(float *data, int n)
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
    
/*--------------- _convolveImageHoriz ---------------*/

typedef struct {
    float *data;   
    int width;
} ConvolutionKernel;

//KERNEL
__global__ void convolve_horiz_kernel(const float* imgin, const float* kernel_data, float* imgout, int ncols, int nrows, int kernelWidth)
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

/*--------------- trackFeatures.c Functions (GPU wrappers) ---------------*/
//typedef float* _FloatWindow; /* same as _FloatWindow in trackFeatures.c */

// Device inline bilinear sampler (matches CPU _interpolate math)
__device__ __inline__ float sample_bilinear(const float* img, int ncols, int nrows, float x, float y)
{
    int xt = (int)x;
    int yt = (int)y;
    float ax = x - xt;
    float ay = y - yt;

    // No checks here: assume caller provided valid coords as in the CPU version.
    int base = yt * ncols + xt;
    float v00 = img[base];
    float v10 = img[base + 1];
    float v01 = img[base + ncols];
    float v11 = img[base + ncols + 1];

    float omx = 1.0f - ax;
    float omy = 1.0f - ay;

    return omx * omy * v00 + ax * omy * v10 + omx * ay * v01 + ax * ay * v11;
}

//////////////////////////////////////////////////////////////////////////////
// 1) _computeIntensityDifference  -> gpu_computeIntensityDifference
//////////////////////////////////////////////////////////////////////////////

__global__
static void k_computeIntensityDifference(const float* img1, const float* img2, int ncols, int nrows,
                                         float x1, float y1, float x2, float y2,
                                         int width, int height, float* out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int N = width * height;
    if (tid >= N) return;

    int hw = width / 2;
    int hh = height / 2;

    int local_j = tid / width;           // 0..height-1
    int local_i = tid % width;           // 0..width-1
    int i = local_i - hw;                // -hw .. hw
    int j = local_j - hh;                // -hh .. hh

    float sx1 = x1 + (float)i;
    float sy1 = y1 + (float)j;
    float sx2 = x2 + (float)i;
    float sy2 = y2 + (float)j;

    float g1 = sample_bilinear(img1, ncols, nrows, sx1, sy1);
    float g2 = sample_bilinear(img2, ncols, nrows, sx2, sy2);
    out[tid] = g1 - g2;
}

 
void gpu_computeIntensityDifference(_KLT_FloatImage img1,
                                     _KLT_FloatImage img2,
                                     float x1, float y1,
                                     float x2, float y2,
                                     int width, int height,
                                     _FloatWindow imgdiff)
{
    int ncols = img1->ncols, nrows = img1->nrows;
    int N = width * height;
    size_t imgBytes = (size_t)ncols * nrows * sizeof(float);
    size_t outBytes = (size_t)N * sizeof(float);

    float *d_img1 = NULL, *d_img2 = NULL, *d_out = NULL;

    CUDA_CHECK(cudaMalloc(&d_img1, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_img2, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_out, outBytes));

    CUDA_CHECK(cudaMemcpy(d_img1, img1->data, imgBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_img2, img2->data, imgBytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    k_computeIntensityDifference<<<blocks, threads>>>(d_img1, d_img2, ncols, nrows,
                                                      x1, y1, x2, y2, width, height, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(imgdiff, d_out, outBytes, cudaMemcpyDeviceToHost));

    cudaFree(d_img1);
    cudaFree(d_img2);
    cudaFree(d_out);
}

//////////////////////////////////////////////////////////////////////////////
// 2) _computeGradientSum -> gpu_computeGradientSum
//////////////////////////////////////////////////////////////////////////////

__global__
static void k_computeGradientSum(const float* gradx1, const float* grady1,
                                 const float* gradx2, const float* grady2,
                                 int ncols, int nrows,
                                 float x1, float y1, float x2, float y2,
                                 int width, int height,
                                 float* out_gradx, float* out_grady)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int N = width * height;
    if (tid >= N) return;

    int hw = width / 2;
    int hh = height / 2;

    int local_j = tid / width;
    int local_i = tid % width;
    int i = local_i - hw;
    int j = local_j - hh;

    float sx1 = x1 + (float)i;
    float sy1 = y1 + (float)j;
    float sx2 = x2 + (float)i;
    float sy2 = y2 + (float)j;

    float gx1 = sample_bilinear(gradx1, ncols, nrows, sx1, sy1);
    float gx2 = sample_bilinear(gradx2, ncols, nrows, sx2, sy2);
    out_gradx[tid] = gx1 + gx2;

    float gy1 = sample_bilinear(grady1, ncols, nrows, sx1, sy1);
    float gy2 = sample_bilinear(grady2, ncols, nrows, sx2, sy2);
    out_grady[tid] = gy1 + gy2;
}

 
void gpu_computeGradientSum(_KLT_FloatImage gradx1,
                            _KLT_FloatImage grady1,
                            _KLT_FloatImage gradx2,
                            _KLT_FloatImage grady2,
                            float x1, float y1,
                            float x2, float y2,
                            int width, int height,
                            _FloatWindow gradx,
                            _FloatWindow grady)
{
    int ncols = gradx1->ncols, nrows = gradx1->nrows;
    int N = width * height;
    size_t imgBytes = (size_t)ncols * nrows * sizeof(float);
    size_t outBytes = (size_t)N * sizeof(float);

    float *d_gx1 = NULL, *d_gy1 = NULL, *d_gx2 = NULL, *d_gy2 = NULL, *d_outgx = NULL, *d_outgy = NULL;

    CUDA_CHECK(cudaMalloc(&d_gx1, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_gy1, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_gx2, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_gy2, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_outgx, outBytes));
    CUDA_CHECK(cudaMalloc(&d_outgy, outBytes));

    CUDA_CHECK(cudaMemcpy(d_gx1, gradx1->data, imgBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gy1, grady1->data, imgBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gx2, gradx2->data, imgBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gy2, grady2->data, imgBytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    k_computeGradientSum<<<blocks, threads>>>(d_gx1, d_gy1, d_gx2, d_gy2,
                                              ncols, nrows, x1, y1, x2, y2,
                                              width, height, d_outgx, d_outgy);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(gradx, d_outgx, outBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(grady, d_outgy, outBytes, cudaMemcpyDeviceToHost));

    cudaFree(d_gx1); cudaFree(d_gy1); cudaFree(d_gx2); cudaFree(d_gy2);
    cudaFree(d_outgx); cudaFree(d_outgy);
}

//////////////////////////////////////////////////////////////////////////////
// 3) _computeIntensityDifferenceLightingInsensitive
//    -> First kernel: sum and sumSq via atomic adds
//    -> Host computes alpha/belta
//    -> Second kernel: compute final pixels
//////////////////////////////////////////////////////////////////////////////

__global__
static void k_idli_sum(const float* img1, const float* img2, int ncols, int nrows,
                       float x1, float y1, float x2, float y2, int width, int height,
                       float* sum1, float* sum2, float* sum1sq, float* sum2sq)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int N = width * height;
    if (tid >= N) return;

    int hw = width / 2;
    int hh = height / 2;
    int local_j = tid / width;
    int local_i = tid % width;
    int i = local_i - hw;
    int j = local_j - hh;

    float sx1 = x1 + (float)i;
    float sy1 = y1 + (float)j;
    float sx2 = x2 + (float)i;
    float sy2 = y2 + (float)j;

    float g1 = sample_bilinear(img1, ncols, nrows, sx1, sy1);
    float g2 = sample_bilinear(img2, ncols, nrows, sx2, sy2);

    // atomic adds into single device scalars
    atomicAdd(sum1, g1);
    atomicAdd(sum2, g2);
    atomicAdd(sum1sq, g1 * g1);
    atomicAdd(sum2sq, g2 * g2);
}

__global__
static void k_idli_final(const float* img1, const float* img2, int ncols, int nrows,
                         float x1, float y1, float x2, float y2, int width, int height,
                         float alpha, float belta, float* out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int N = width * height;
    if (tid >= N) return;

    int hw = width / 2;
    int hh = height / 2;
    int local_j = tid / width;
    int local_i = tid % width;
    int i = local_i - hw;
    int j = local_j - hh;

    float sx1 = x1 + (float)i;
    float sy1 = y1 + (float)j;
    float sx2 = x2 + (float)i;
    float sy2 = y2 + (float)j;

    float g1 = sample_bilinear(img1, ncols, nrows, sx1, sy1);
    float g2 = sample_bilinear(img2, ncols, nrows, sx2, sy2);

    out[tid] = g1 - g2 * alpha - belta;
}

 
void gpu_computeIntensityDifferenceLightingInsensitive(_KLT_FloatImage img1,
                                                       _KLT_FloatImage img2,
                                                       float x1, float y1,
                                                       float x2, float y2,
                                                       int width, int height,
                                                       _FloatWindow imgdiff)
{
    int ncols = img1->ncols, nrows = img1->nrows;
    int N = width * height;
    size_t imgBytes = (size_t)ncols * nrows * sizeof(float);
    size_t outBytes = (size_t)N * sizeof(float);

    float *d_img1 = NULL, *d_img2 = NULL;
    float *d_out = NULL;
    float *d_sum1 = NULL, *d_sum2 = NULL, *d_sum1sq = NULL, *d_sum2sq = NULL;

    CUDA_CHECK(cudaMalloc(&d_img1, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_img2, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_out, outBytes));
    CUDA_CHECK(cudaMalloc(&d_sum1, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum2, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum1sq, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum2sq, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_img1, img1->data, imgBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_img2, img2->data, imgBytes, cudaMemcpyHostToDevice));
    // zero sums
    CUDA_CHECK(cudaMemset(d_sum1, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sum2, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sum1sq, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sum2sq, 0, sizeof(float)));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    k_idli_sum<<<blocks, threads>>>(d_img1, d_img2, ncols, nrows, x1, y1, x2, y2, width, height,
                                    d_sum1, d_sum2, d_sum1sq, d_sum2sq);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // copy sums back to host
    float sum1 = 0.0f, sum2 = 0.0f, sum1sq = 0.0f, sum2sq = 0.0f;
    CUDA_CHECK(cudaMemcpy(&sum1, d_sum1, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sum2, d_sum2, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sum1sq, d_sum1sq, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sum2sq, d_sum2sq, sizeof(float), cudaMemcpyDeviceToHost));

    float mean1 = sum1sq / (float)N;
    float mean2 = sum2sq / (float)N;
    float alpha = sqrtf(mean1 / mean2);
    mean1 = sum1 / (float)N;
    mean2 = sum2 / (float)N;
    float belta = mean1 - alpha * mean2;

    // final pass
    k_idli_final<<<blocks, threads>>>(d_img1, d_img2, ncols, nrows, x1, y1, x2, y2, width, height,
                                      alpha, belta, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(imgdiff, d_out, outBytes, cudaMemcpyDeviceToHost));

    // free
    cudaFree(d_img1); cudaFree(d_img2); cudaFree(d_out);
    cudaFree(d_sum1); cudaFree(d_sum2); cudaFree(d_sum1sq); cudaFree(d_sum2sq);
}

//////////////////////////////////////////////////////////////////////////////
// 4) _computeGradientSumLightingInsensitive
//    same pattern: reduce sums for img1/img2 -> alpha -> final pass
//////////////////////////////////////////////////////////////////////////////

__global__
static void k_gsli_sum(const float* img1, const float* img2, int ncols, int nrows,
                       float x1, float y1, float x2, float y2, int width, int height,
                       float* sum1, float* sum2)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int N = width * height;
    if (tid >= N) return;

    int hw = width / 2;
    int hh = height / 2;
    int local_j = tid / width;
    int local_i = tid % width;
    int i = local_i - hw;
    int j = local_j - hh;

    float sx1 = x1 + (float)i;
    float sy1 = y1 + (float)j;
    float sx2 = x2 + (float)i;
    float sy2 = y2 + (float)j;

    float g1 = sample_bilinear(img1, ncols, nrows, sx1, sy1);
    float g2 = sample_bilinear(img2, ncols, nrows, sx2, sy2);

    atomicAdd(sum1, g1);
    atomicAdd(sum2, g2);
}

__global__
static void k_gsli_final(const float* gx1, const float* gx2, const float* gy1, const float* gy2,
                         int ncols, int nrows, float x1, float y1, float x2, float y2,
                         int width, int height, float alpha, float* out_gx, float* out_gy)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int N = width * height;
    if (tid >= N) return;

    int hw = width / 2;
    int hh = height / 2;
    int local_j = tid / width;
    int local_i = tid % width;
    int i = local_i - hw;
    int j = local_j - hh;

    float sx1 = x1 + (float)i;
    float sy1 = y1 + (float)j;
    float sx2 = x2 + (float)i;
    float sy2 = y2 + (float)j;

    float g1x = sample_bilinear(gx1, ncols, nrows, sx1, sy1);
    float g2x = sample_bilinear(gx2, ncols, nrows, sx2, sy2);
    out_gx[tid] = g1x + g2x * alpha;

    float g1y = sample_bilinear(gy1, ncols, nrows, sx1, sy1);
    float g2y = sample_bilinear(gy2, ncols, nrows, sx2, sy2);
    out_gy[tid] = g1y + g2y * alpha;
}

 
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
                                               _FloatWindow grady)
{
    int ncols = img1->ncols, nrows = img1->nrows;
    int N = width * height;
    size_t imgBytes = (size_t)ncols * nrows * sizeof(float);
    size_t outBytes = (size_t)N * sizeof(float);

    float *d_img1 = NULL, *d_img2 = NULL;
    float *d_sum1 = NULL, *d_sum2 = NULL;
    float *d_outgx = NULL, *d_outgy = NULL;

    CUDA_CHECK(cudaMalloc(&d_img1, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_img2, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_sum1, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum2, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_outgx, outBytes));
    CUDA_CHECK(cudaMalloc(&d_outgy, outBytes));

    CUDA_CHECK(cudaMemcpy(d_img1, img1->data, imgBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_img2, img2->data, imgBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_sum1, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sum2, 0, sizeof(float)));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // sum phase
    k_gsli_sum<<<blocks, threads>>>(d_img1, d_img2, ncols, nrows, x1, y1, x2, y2, width, height, d_sum1, d_sum2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float sum1 = 0.0f, sum2 = 0.0f;
    CUDA_CHECK(cudaMemcpy(&sum1, d_sum1, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sum2, d_sum2, sizeof(float), cudaMemcpyDeviceToHost));

    float mean1 = sum1 / (float)N;
    float mean2 = sum2 / (float)N;
    float alpha = sqrtf(mean1 / mean2);

    // final phase: use gradx1/grady1/gradx2/grady2 device copies
    // copy gradient images to device
    float *d_gx1 = NULL, *d_gy1 = NULL, *d_gx2 = NULL, *d_gy2 = NULL;
    CUDA_CHECK(cudaMalloc(&d_gx1, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_gy1, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_gx2, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_gy2, imgBytes));

    CUDA_CHECK(cudaMemcpy(d_gx1, gradx1->data, imgBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gy1, grady1->data, imgBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gx2, gradx2->data, imgBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gy2, grady2->data, imgBytes, cudaMemcpyHostToDevice));

    k_gsli_final<<<blocks, threads>>>(d_gx1, d_gx2, d_gy1, d_gy2, ncols, nrows,
                                      x1, y1, x2, y2, width, height, alpha, d_outgx, d_outgy);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(gradx, d_outgx, outBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(grady, d_outgy, outBytes, cudaMemcpyDeviceToHost));

    // free
    cudaFree(d_img1); cudaFree(d_img2); cudaFree(d_sum1); cudaFree(d_sum2);
    cudaFree(d_outgx); cudaFree(d_outgy);
    cudaFree(d_gx1); cudaFree(d_gy1); cudaFree(d_gx2); cudaFree(d_gy2);
}

//////////////////////////////////////////////////////////////////////////////
// 5) _am_getGradientWinAffine -> gpu_am_getGradientWinAffine
//////////////////////////////////////////////////////////////////////////////

__global__
static void k_am_getGradientWinAffine(const float* gx, const float* gy,
                                      int ncols, int nrows,
                                      float xc, float yc,
                                      float Axx, float Ayx, float Axy, float Ayy,
                                      int width, int height,
                                      float* out_gx, float* out_gy)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int N = width * height;
    if (tid >= N) return;

    int hw = width / 2, hh = height / 2;
    int local_j = tid / width;
    int local_i = tid % width;
    int i = local_i - hw;
    int j = local_j - hh;

    float mi = Axx * i + Axy * j;
    float mj = Ayx * i + Ayy * j;
    float sx = xc + mi;
    float sy = yc + mj;

    out_gx[tid] = sample_bilinear(gx, ncols, nrows, sx, sy);
    out_gy[tid] = sample_bilinear(gy, ncols, nrows, sx, sy);
}

 
void gpu_am_getGradientWinAffine(_KLT_FloatImage in_gradx,
                                 _KLT_FloatImage in_grady,
                                 float x, float y,
                                 float Axx, float Ayx, float Axy, float Ayy,
                                 int width, int height,
                                 _FloatWindow out_gradx,
                                 _FloatWindow out_grady)
{
    int ncols = in_gradx->ncols, nrows = in_gradx->nrows;
    int N = width * height;
    size_t imgBytes = (size_t)ncols * nrows * sizeof(float);
    size_t outBytes = (size_t)N * sizeof(float);

    float *d_gx = NULL, *d_gy = NULL, *d_outgx = NULL, *d_outgy = NULL;
    CUDA_CHECK(cudaMalloc(&d_gx, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_gy, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_outgx, outBytes));
    CUDA_CHECK(cudaMalloc(&d_outgy, outBytes));

    CUDA_CHECK(cudaMemcpy(d_gx, in_gradx->data, imgBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gy, in_grady->data, imgBytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    k_am_getGradientWinAffine<<<blocks, threads>>>(d_gx, d_gy, ncols, nrows, x, y, Axx, Ayx, Axy, Ayy, width, height, d_outgx, d_outgy);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(out_gradx, d_outgx, outBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out_grady, d_outgy, outBytes, cudaMemcpyDeviceToHost));

    cudaFree(d_gx); cudaFree(d_gy); cudaFree(d_outgx); cudaFree(d_outgy);
}

//////////////////////////////////////////////////////////////////////////////
// 6) _am_computeAffineMappedImage -> gpu_am_computeAffineMappedImage
//////////////////////////////////////////////////////////////////////////////

__global__
static void k_am_computeAffineMappedImage(const float* img, int ncols, int nrows,
                                          float xc, float yc,
                                          float Axx, float Ayx, float Axy, float Ayy,
                                          int width, int height, float* out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int N = width * height;
    if (tid >= N) return;

    int hw = width / 2, hh = height / 2;
    int local_j = tid / width;
    int local_i = tid % width;
    int i = local_i - hw;
    int j = local_j - hh;

    float mi = Axx * i + Axy * j;
    float mj = Ayx * i + Ayy * j;
    float sx = xc + mi;
    float sy = yc + mj;

    out[tid] = sample_bilinear(img, ncols, nrows, sx, sy);
}

 
void gpu_am_computeAffineMappedImage(_KLT_FloatImage img,
                                     float x, float y,
                                     float Axx, float Ayx, float Axy, float Ayy,
                                     int width, int height,
                                     _FloatWindow imgdiff)
{
    int ncols = img->ncols, nrows = img->nrows;
    int N = width * height;
    size_t imgBytes = (size_t)ncols * nrows * sizeof(float);
    size_t outBytes = (size_t)N * sizeof(float);

    float *d_img = NULL, *d_out = NULL;
    CUDA_CHECK(cudaMalloc(&d_img, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_out, outBytes));

    CUDA_CHECK(cudaMemcpy(d_img, img->data, imgBytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    k_am_computeAffineMappedImage<<<blocks, threads>>>(d_img, ncols, nrows, x, y, Axx, Ayx, Axy, Ayy, width, height, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(imgdiff, d_out, outBytes, cudaMemcpyDeviceToHost));

    cudaFree(d_img); cudaFree(d_out);
}

//////////////////////////////////////////////////////////////////////////////
// 7) _am_computeIntensityDifferenceAffine -> gpu_am_computeIntensityDifferenceAffine
//////////////////////////////////////////////////////////////////////////////

__global__
static void k_am_computeIntensityDifferenceAffine(const float* img1, const float* img2,
                                                  int ncols, int nrows,
                                                  float x1, float y1, float x2, float y2,
                                                  float Axx, float Ayx, float Axy, float Ayy,
                                                  int width, int height, float* out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int N = width * height;
    if (tid >= N) return;

    int hw = width / 2, hh = height / 2;
    int local_j = tid / width;
    int local_i = tid % width;
    int i = local_i - hw;
    int j = local_j - hh;

    float g1 = sample_bilinear(img1, ncols, nrows, x1 + (float)i, y1 + (float)j);

    float mi = Axx * i + Axy * j;
    float mj = Ayx * i + Ayy * j;
    float sx2 = x2 + mi;
    float sy2 = y2 + mj;

    float g2 = sample_bilinear(img2, ncols, nrows, sx2, sy2);
    out[tid] = g1 - g2;
}

 
void gpu_am_computeIntensityDifferenceAffine(_KLT_FloatImage img1,
                                             _KLT_FloatImage img2,
                                             float x1, float y1,
                                             float x2, float y2,
                                             float Axx, float Ayx, float Axy, float Ayy,
                                             int width, int height,
                                             _FloatWindow imgdiff)
{
    int ncols = img1->ncols, nrows = img1->nrows;
    int N = width * height;
    size_t imgBytes = (size_t)ncols * nrows * sizeof(float);
    size_t outBytes = (size_t)N * sizeof(float);

    float *d_img1 = NULL, *d_img2 = NULL, *d_out = NULL;
    CUDA_CHECK(cudaMalloc(&d_img1, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_img2, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_out, outBytes));

    CUDA_CHECK(cudaMemcpy(d_img1, img1->data, imgBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_img2, img2->data, imgBytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    k_am_computeIntensityDifferenceAffine<<<blocks, threads>>>(d_img1, d_img2, ncols, nrows,
                                                               x1, y1, x2, y2,
                                                               Axx, Ayx, Axy, Ayy,
                                                               width, height, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(imgdiff, d_out, outBytes, cudaMemcpyDeviceToHost));

    cudaFree(d_img1); cudaFree(d_img2); cudaFree(d_out);
}

