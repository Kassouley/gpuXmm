#ifndef __GPUXMM_CUDA_H
#define __GPUXMM_CUDA_H
#include <cuda_runtime.h>

#define CHECK(cmd) \
{\
    cudaError_t error  = cmd;\
    if (error != cudaSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", cudaGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
    }\
}

#if defined(CUBLAS) || defined(CUBLAS_WO_DT) 
#include <cublas_v2.h>
typedef cublasHandle_t gpuXmm_handle_t;

#define __GPUXMM_NEED_HANDLE

#define gpuXmm_handle_create(handle) \
{\
    gpuXmmtx_rangePush("gpuXmmtx_cublasCreate"); \
    cublasCreate(&handle);\
    gpuXmmtx_rangePop(); \
}

#define gpuXmm_handle_destroy(handle) \
{\
    gpuXmmtx_rangePush("gpuXmmtx_cublasDestroy"); \
    cublasDestroy(handle);\
    gpuXmmtx_rangePop(); \
}

#endif

#define gpuXmm_malloc(ptr, size) \
{\
    gpuXmmtx_rangePush("gpuXmmtx_cudaMalloc"); \
    CHECK(cudaMalloc(&ptr, size));\
    gpuXmmtx_rangePop(); \
}

#define gpuXmm_free(ptr) \
{\
    gpuXmmtx_rangePush("gpuXmmtx_cudaFree"); \
    CHECK(cudaFree(ptr));\
    gpuXmmtx_rangePop(); \
}

#define gpuXmm_memcpy_HtD(dst, src, size) \
{\
    gpuXmmtx_rangePush("gpuXmmtx_cudaMemcpy_HtD"); \
    CHECK(cudaMemcpy(dst, src, size,cudaMemcpyHostToDevice));\
    gpuXmmtx_rangePop(); \
}

#define gpuXmm_memcpy_DtH(dst, src, size) \
{\
    gpuXmmtx_rangePush("gpuXmmtx_cudaMemcpy_DtH"); \
    CHECK(cudaMemcpy(dst, src, size,cudaMemcpyDeviceToHost));\
    gpuXmmtx_rangePop(); \
}

#define gpuXmm_deviceSynchronize() \
{\
    gpuXmmtx_rangePush("gpuXmmtx_cudaDeviceSynchronize"); \
    cudaDeviceSynchronize();\
    gpuXmmtx_rangePop(); \
}

#endif