#include <stdio.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include "gpuXmm.h"
#include "kernel.h"

#ifdef ROCBLAS
void kernel_gpuXmm (rocblas_handle handle, unsigned int m, unsigned int n, unsigned int p, 
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

    const gpuXmm_precision_t alpha = 1.0; 
    const gpuXmm_precision_t beta = 0.0;

    #ifdef SP
        rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                p, m, n, &alpha, d_b, p, d_a, n, &beta, d_c, p); 
    #else // DP
        rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none,
                p, m, n, &alpha, d_b, p, d_a, n, &beta, d_c, p); 
    #endif

    hipMemcpy(c, d_c, size_c, hipMemcpyDeviceToHost);

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
}
#endif

#ifdef ROCBLAS_WO_DT

void kernel_gpuXmm (rocblas_handle handle, unsigned int m, unsigned int n, unsigned int p, 
                    const gpuXmm_precision_t* a, const gpuXmm_precision_t* b, gpuXmm_precision_t* c)
{
    const gpuXmm_precision_t alpha = 1.0f; 
    const gpuXmm_precision_t beta = 0.0f; 
    
    #ifdef SP
        rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                p, m, n, &alpha, b, p, a, n, &beta, c, p); 
    #else // DP
        rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none,
                p, m, n, &alpha, b, p, a, n, &beta, c, p); 
    #endif
}
#endif

