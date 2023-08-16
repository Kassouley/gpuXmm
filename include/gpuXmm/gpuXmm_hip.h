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
            rocblas_create_handle(&handle);\
        }

        #define gpuXmm_handle_destroy(handle) \
        {\
            rocblas_destroy_handle(handle);\
        }
    #endif

    #define gpuXmm_malloc(ptr, size) \
    {\
        CHECK(hipMalloc((void**)&ptr, size));\
    }

    #define gpuXmm_free(ptr) \
    {\
        CHECK(hipFree(ptr));\
    }

    #define gpuXmm_memcpy_HtD(dst, src, size) \
    {\
        CHECK(hipMemcpy(dst, src, size, hipMemcpyHostToDevice));\
    }

    #define gpuXmm_memcpy_DtH(dst, src, size) \
    {\
        CHECK(hipMemcpy(dst, src, size, hipMemcpyDeviceToHost));\
    }

    #define gpuXmm_deviceSynchronize() \
    {\
        hipDeviceSynchronize();\
    }

#endif