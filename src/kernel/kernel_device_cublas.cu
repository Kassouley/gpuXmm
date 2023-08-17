#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "gpuXmm.h"
#include "kernel.h"

#ifdef CUBLAS
void kernel_gpuXmm (cublasHandle_t handle, unsigned int m, unsigned int n, unsigned int p, 
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


    gpuXmm_precision_t alpha = 1.0f;
    gpuXmm_precision_t beta = 0.0f;
   
    #ifdef SP
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                p, m, n, &alpha, d_b, p, d_a, n, &beta, d_c, p);
    #else // DP
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                p, m, n, &alpha, d_b, p, d_a, n, &beta, d_c, p);
    #endif 
   
    CHECK(cudaMemcpy(c, d_c, size_c, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
}
#endif


#ifdef CUBLAS_WO_DT
void kernel_gpuXmm (cublasHandle_t handle, unsigned int m, unsigned int n, unsigned int p, 
                    const gpuXmm_precision_t* a, const gpuXmm_precision_t* b, gpuXmm_precision_t* c)
{ 
    gpuXmm_precision_t alpha = 1.0f;
    gpuXmm_precision_t beta = 0.0f;

    #ifdef SP
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                p, m, n, &alpha, b, p, a, n, &beta, c, p);
    #else // DP
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                p, m, n, &alpha, b, p, a, n, &beta, c, p);
    #endif
}
#endif
