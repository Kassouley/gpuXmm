#ifndef __GPUXMMTX_H
#define __GPUXMMTX_H

#ifdef USETX    
    #ifdef AMD
        #include <roctracer/roctx.h>

        #define gpuXmmtx_rangePush(name) \
        { \
            roctxRangePush(name); \
        }
        #define gpuXmmtx_rangePop() \
        { \
            roctxRangePop(); \
        }
    #endif

    #ifdef NVIDIA
        #include <nvToolsExt.h>

        #define gpuXmmtx_rangePush(name) \
        { \
            nvtxRangePush(name); \
        }
        #define gpuXmmtx_rangePop() \
        { \
            nvtxRangePop(); \
        }
    #endif
#else
    #define gpuXmmtx_rangePush(name)
    #define gpuXmmtx_rangePop()
#endif
#endif