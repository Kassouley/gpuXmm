#ifndef __GPUXMM_ACC_H
#define __GPUXMM_ACC_H
#include <openacc.h>

#define gpuXmm_malloc(ptr, size) \
{\
    ptr = acc_malloc(size); \
    if ( ptr == NULL ) \
    { \
        fprintf(stderr, "error: 'malloc ptr is null' at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
    } \
}

#define gpuXmm_free(ptr) \
{\
    acc_free(ptr); \
}

#define gpuXmm_memcpy_HtD(dst,src,size) \
{\
    acc_memcpy_to_device(dst, src, size); \
}

#define gpuXmm_memcpy_DtH(dst,src,size) \
{\
    acc_memcpy_from_device(dst,src,size); \
}
#endif