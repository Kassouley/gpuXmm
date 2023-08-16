#ifndef __KERNEL_H
#define __KERNEL_H

#if defined(HIP) || defined(CUDA)
__global__ void kernel_gpuXmm_aux (unsigned int m, unsigned int n, unsigned int p, 
                    const gpuXmm_precision_t* a, const gpuXmm_precision_t* b, gpuXmm_precision_t* c);
#endif

#if defined(ROCBLAS_WO_DT) || defined(ROCBLAS)
    #include <rocblas/rocblas.h>
    void kernel_gpuXmm (rocblas_handle handle, unsigned int m, unsigned int n, unsigned int p, 
                        const gpuXmm_precision_t* a, const gpuXmm_precision_t* b, gpuXmm_precision_t* c);                            

#elif defined(CUBLAS_WO_DT) || defined(CUBLAS)
    #include <cublas_v2.h>
    void kernel_gpuXmm (cublasHandle_t handle, unsigned int m, unsigned int n, unsigned int p, 
                        const gpuXmm_precision_t* a, const gpuXmm_precision_t* b, gpuXmm_precision_t* c);

    #ifdef SP
    #define kernel_cublas_Xgemm(handle, transA, transB, m, n, p, alpha, a, ldA, b, ldB, beta, c, ldC) \
    { \
        cublasSgemm(handle, transA, transB, m, p, n, &alpha, a, ldA, b, ldB, &beta, c, ldC); \
    }
    #else // DP
    #define kernel_cublas_Xgemm(handle, transA, transB, m, n, p, alpha, a, ldA, b, ldB, beta, c, ldC) \
    { \
        cublasDgemm(handle, transA, transB, m, p, n, &alpha, a, ldA, b, ldB, &beta, c, ldC); \
    }
    #endif

#else
    void kernel_gpuXmm (unsigned int m, unsigned int n, unsigned int p, 
                        const gpuXmm_precision_t* a, const gpuXmm_precision_t* b, gpuXmm_precision_t* c);
#endif

#endif

