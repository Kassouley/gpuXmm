#ifndef __GPUXMM_OMP_H
#define __GPUXMM_OMP_H
#include <omp.h>

#define gpuXmm_malloc(ptr, size) \
{\
    ptr = omp_target_alloc(size, 0);\
    if ( ptr == NULL ) \
    { \
        fprintf(stderr, "error: 'malloc ptr is null' at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
    } \
}

#define gpuXmm_free(ptr) \
{\
    omp_target_free(ptr, 0);\
}

#define gpuXmm_memcpy_HtD(dst,src,size) \
{\
    omp_target_memcpy(dst, src, size, 0, 0, 0, omp_get_initial_device());\
}

#define gpuXmm_memcpy_DtH(dst,src,size) \
{\
    omp_target_memcpy(dst, src, size, 0, 0, omp_get_initial_device(), 0);\
}
#endif
