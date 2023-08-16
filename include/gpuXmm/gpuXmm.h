#ifndef GPUBLAS_H
#define GPUBLAS_H
    #include "gpuXmm_precision.h"

    #if defined(HIP) || defined(HIP_WO_DT) || defined(ROCBLAS) || defined(ROCBLAS_WO_DT)
    #include "gpuXmm_hip.h"
    

    #endif
    #if defined(CUDA) || defined(CUDA_WO_DT) || defined(CUBLAS) || defined(CUBLAS_WO_DT)
    #include "gpuXmm_cuda.h"
        

    #endif
    #if defined(GPU_OMP) || defined(GPU_OMP_WO_DT)
    #include "gpuXmm_omp.h"
        

    #endif
    #if defined(ACC) || defined(ACC_WO_DT)
    #include "gpuXmm_acc.h"
        

    #endif

#endif