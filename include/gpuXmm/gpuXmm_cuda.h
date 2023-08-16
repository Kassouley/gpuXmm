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
    cublasCreate(&handle);\
}

#define gpuXmm_handle_destroy(handle) \
{\
    cublasDestroy(handle);\
}

#endif

#define gpuXmm_malloc(ptr, size) \
{\
    CHECK(cudaMalloc(&ptr, size));\
}

#define gpuXmm_free(ptr) \
{\
    CHECK(cudaFree(ptr));\
}

#define gpuXmm_memcpy_HtD(dst, src, size) \
{\
    CHECK(cudaMemcpy(dst, src, size,cudaMemcpyHostToDevice));\
}

#define gpuXmm_memcpy_DtH(dst, src, size) \
{\
    CHECK(cudaMemcpy(dst, src, size,cudaMemcpyDeviceToHost));\
}

#define gpuXmm_deviceSynchronize() \
{\
    cudaDeviceSynchronize();\
}

#endif