#ifndef __GPUXMM_OMP_H
#define __GPUXMM_OMP_H
#include <omp.h>

#define gpuXmm_malloc(ptr, size) \
{\
    gpuXmmtx_rangePush("gpuXmmtx_omp_target_alloc"); \
    ptr = omp_target_alloc(size, 0);\
    if ( ptr == NULL ) \
    { \
        fprintf(stderr, "error: 'malloc ptr is null' at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
    } \
    gpuXmmtx_rangePop(); \
}

#define gpuXmm_free(ptr) \
{\
    gpuXmmtx_rangePush("gpuXmmtx_omp_target_free"); \
    omp_target_free(ptr, 0);\
    gpuXmmtx_rangePop(); \
}

#define gpuXmm_memcpy_HtD(dst,src,size) \
{\
    gpuXmmtx_rangePush("gpuXmmtx_omp_target_memcpy_HtD"); \
    omp_target_memcpy(dst, src, size, 0, 0, 0, omp_get_initial_device());\
    gpuXmmtx_rangePop(); \
}

#define gpuXmm_memcpy_DtH(dst,src,size) \
{\
    gpuXmmtx_rangePush("gpuXmmtx_omp_target_memcpy_DtH"); \
    omp_target_memcpy(dst, src, size, 0, 0, omp_get_initial_device(), 0);\
    gpuXmmtx_rangePop(); \
}
#endif
