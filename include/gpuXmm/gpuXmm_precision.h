#ifndef __GPUXMM_PRECISION_H
#define __GPUXMM_PRECISION_H

#ifdef SP
    typedef float gpuXmm_precision_t;
#else // DP
    typedef double gpuXmm_precision_t;
#endif

#endif