#include <stdio.h>
#include <cuda_runtime.h>
#include "gpuXmm.h"
#include "kernel.h"

__global__ void kernel_gpuXmm_aux (unsigned int m, unsigned int n, unsigned int p, 
                                    const gpuXmm_precision_t* a, const gpuXmm_precision_t* b, gpuXmm_precision_t* c)
{ 
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < p) 
    {
        gpuXmm_precision_t sum = 0.0f;
        for (unsigned int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * p + col];
        }
        c[row * p + col] = sum;
    }
}

#ifdef CUDA
void kernel_gpuXmm (unsigned int m, unsigned int n, unsigned int p, 
                    const gpuXmm_precision_t* a, const gpuXmm_precision_t* b, gpuXmm_precision_t* c)
{
    int size_a = m * n * sizeof(gpuXmm_precision_t);
    int size_b = n * p * sizeof(gpuXmm_precision_t);
    int size_c = m * p * sizeof(gpuXmm_precision_t);
    
    gpuXmm_precision_t* d_a;
    gpuXmm_precision_t* d_b;
    gpuXmm_precision_t* d_c;
    
	CHECK(cudaMalloc(&d_a, size_a));
    CHECK(cudaMalloc(&d_b, size_b));
    CHECK(cudaMalloc(&d_c, size_c));

    CHECK(cudaMemcpy(d_a, a, size_a, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b, size_b, cudaMemcpyHostToDevice));

    dim3 blockDim (m, p);
    dim3 gridDim (1,1);
    if ( m > 32 )
    {
        blockDim.x = 32;
        gridDim.x = ceil(double(m)/double(blockDim.x));
    }
    if ( p > 32 )
    {
        blockDim.y = 32;
        gridDim.y = ceil(double(p)/double(blockDim.y));
    }

    kernel_gpuXmm_aux<<<gridDim, blockDim>>>( m, n, p, d_a, d_b, d_c);
        
	CHECK(cudaMemcpy(c, d_c, size_c, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
}
#endif

#ifdef CUDA_WO_DT
void kernel_gpuXmm (unsigned int m, unsigned int n, unsigned int p, 
                    const gpuXmm_precision_t* a, const gpuXmm_precision_t* b, gpuXmm_precision_t* c)
{
    dim3 blockDim (m, p);
    dim3 gridDim (1,1);
    if ( m > 32 )
    {
        blockDim.x = 32;
        gridDim.x = ceil(double(m)/double(blockDim.x));
    }
    if ( p > 32 )
    {
        blockDim.y = 32;
        gridDim.y = ceil(double(p)/double(blockDim.y));
    }
    kernel_gpuXmm_aux<<<gridDim, blockDim>>>( m, n, p, d_a, d_b, d_c);
}
#endif