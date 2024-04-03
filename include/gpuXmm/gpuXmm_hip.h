#ifndef __GPUXMM_HIP_H
    #define __GPUXMM_HIP_H
    #include <hip/hip_runtime.h>

    #define CHECK(cmd) \
    {\
        hipError_t error  = cmd;\
        if (error != hipSuccess) { \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
            exit(EXIT_FAILURE);\
        }\
    }


    #if defined(ROCBLAS) || defined(ROCBLAS_WO_DT) 
        #include <rocblas/rocblas.h>
        typedef rocblas_handle gpuXmm_handle_t;

        #define __GPUXMM_NEED_HANDLE

        #define gpuXmm_handle_create(handle) \
        {\
            gpuXmmtx_rangePush("gpuXmmtx_rocblas_create_handle"); \
            rocblas_create_handle(&handle); \
            gpuXmmtx_rangePop(); \
        }

        #define gpuXmm_handle_destroy(handle) \
        {\
            gpuXmmtx_rangePush("gpuXmmtx_rocblas_destroy_handle"); \
            rocblas_destroy_handle(handle);\
            gpuXmmtx_rangePop(); \
        }
    #endif

    #define gpuXmm_malloc(ptr, size) \
    {\
        gpuXmmtx_rangePush("gpuXmmtx_hipMalloc"); \
        CHECK(hipMalloc((void**)&ptr, size));\
        gpuXmmtx_rangePop(); \
    }

    #define gpuXmm_free(ptr) \
    {\
        gpuXmmtx_rangePush("gpuXmmtx_hipFree"); \
        CHECK(hipFree(ptr));\
        gpuXmmtx_rangePop(); \
    }

    #define gpuXmm_memcpy_HtD(dst, src, size) \
    {\
        gpuXmmtx_rangePush("gpuXmmtx_hipMemcpy_HtD"); \
        CHECK(hipMemcpy(dst, src, size, hipMemcpyHostToDevice));\
        gpuXmmtx_rangePop(); \
    }

    #define gpuXmm_memcpy_DtH(dst, src, size) \
    {\
        gpuXmmtx_rangePush("gpuXmmtx_hipMemcpy_DtH"); \
        CHECK(hipMemcpy(dst, src, size, hipMemcpyDeviceToHost));\
        gpuXmmtx_rangePop(); \
    }

    #define gpuXmm_deviceSynchronize() \
    {\
        gpuXmmtx_rangePush("gpuXmmtx_hipDeviceSynchronize"); \
        CHECK(hipDeviceSynchronize()); \
        gpuXmmtx_rangePop(); \
    }

#endif