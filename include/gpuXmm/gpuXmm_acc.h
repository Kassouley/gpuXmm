#ifndef __GPUXMM_ACC_H
#define __GPUXMM_ACC_H
#include <openacc.h>

#define gpuXmm_malloc(ptr, size) \
{\
    gpuXmmtx_rangePush("gpuXmmtx_acc_malloc"); \
    ptr = acc_malloc(size); \
    if ( ptr == NULL ) \
    { \
        fprintf(stderr, "error: 'malloc ptr is null' at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
    } \
    gpuXmmtx_rangePop(); \
}

#define gpuXmm_free(ptr) \
{\
    gpuXmmtx_rangePush("gpuXmmtx_acc_free"); \
    acc_free(ptr); \
    gpuXmmtx_rangePop(); \
}

#define gpuXmm_memcpy_HtD(dst,src,size) \
{\
    gpuXmmtx_rangePush("gpuXmmtx_acc_memcpy_to_device"); \
    acc_memcpy_to_device(dst, src, size); \
    gpuXmmtx_rangePop(); \
}

#define gpuXmm_memcpy_DtH(dst,src,size) \
{\
    gpuXmmtx_rangePush("gpuXmmtx_acc_memcpy_from_device"); \
    acc_memcpy_from_device(dst,src,size); \
    gpuXmmtx_rangePop(); \
}
#endif