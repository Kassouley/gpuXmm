#include <stdio.h>
#include <hip/hip_runtime.h>
#include "gpuXmm.h"
#include "kernel.h"

__global__ void kernel_gpuXmm_aux (unsigned int m, unsigned int n, unsigned int p, 
                                    const gpuXmm_precision_t* a, const gpuXmm_precision_t* b, gpuXmm_precision_t* c)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
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

#ifdef HIP
void kernel_gpuXmm (unsigned int m, unsigned int n, unsigned int p, 
                    const gpuXmm_precision_t* a, const gpuXmm_precision_t* b, gpuXmm_precision_t* c)
{
    int size_a = m * n * sizeof(gpuXmm_precision_t);
    int size_b = n * p * sizeof(gpuXmm_precision_t);
    int size_c = m * p * sizeof(gpuXmm_precision_t);
    
    gpuXmm_precision_t* d_a;
    gpuXmm_precision_t* d_b;
    gpuXmm_precision_t* d_c;

    hipMalloc((void**)&d_a, size_a);
    hipMalloc((void**)&d_b, size_b);
    hipMalloc((void**)&d_c, size_c);

    hipMemcpy(d_a, a, size_a, hipMemcpyHostToDevice);
    hipMemcpy(d_b, b, size_b, hipMemcpyHostToDevice);

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

    hipLaunchKernelGGL(kernel_gpuXmm_aux, gridDim, blockDim, 0, 0, m, n, p, d_a, d_b, d_c);
        
    hipMemcpy(c, d_c, size_c, hipMemcpyDeviceToHost);

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
}
#endif

#ifdef HIP_WO_DT
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

    hipLaunchKernelGGL(kernel_gpuXmm_aux, gridDim, blockDim, 0, 0, m, n, p, a, b, c);
}
#endif