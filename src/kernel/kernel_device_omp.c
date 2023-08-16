#include <stdio.h>
#include <omp.h>
#include "gpuXmm.h"
#include "kernel.h"

#ifdef GPU_OMP
void kernel_gpuXmm (unsigned int m, unsigned int n, unsigned int p, 
                    const gpuXmm_precision_t* a, const gpuXmm_precision_t* b, gpuXmm_precision_t* c)
{
    #pragma omp target map(to: a[0:m*n], b[0:n*p]) map(from: c[0:m*p])
    {
        #pragma omp teams distribute parallel for simd collapse(2) 
        for(unsigned int i = 0; i < m; i++)
        {
            for(unsigned int j = 0; j < p; j++)
            {
                c[i*p+j] = 0;
                for(unsigned int k = 0; k < n; k++)
                {
                    c[i*p+j] += a[i*n+k] * b[k*p+j];
                }
            }
        }
    }
}
#endif
#ifdef GPU_OMP_WO_DT
void kernel_gpuXmm (unsigned int m, unsigned int n, unsigned int p, 
                    const gpuXmm_precision_t* a, const gpuXmm_precision_t* b, gpuXmm_precision_t* c)
{
    #pragma omp target is_device_ptr(a, b, c)
    {
        #pragma omp teams distribute parallel for simd collapse(2) 
        for(unsigned int i = 0; i < m; i++)
        {
            for(unsigned int j = 0; j < p; j++)
            {
                c[i*p+j] = 0;
                for(unsigned int k = 0; k < n; k++)
                {
                    c[i*p+j] += a[i*n+k] * b[k*p+j];
                }
            }
        }
    }
}
#endif