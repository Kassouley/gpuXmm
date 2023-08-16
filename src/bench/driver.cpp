#include <stdio.h>
#include <stdlib.h>
#include "gpuXmm.h"
#include "kernel.h"

extern "C" {
#include "array.h"
#include "print_measure.h"
#include "time_measure.h"
}

#define NB_META 31

int main(int argc, char* argv[])
{
    unsigned int m = 0, n = 0, p = 0;
    unsigned int nwu, nrep;
    if (argc != 6) 
    {
        fprintf (stderr, "Usage: %s <m> <n> <p> <nb warmup> <nb rep>\n", argv[0]);
        return 1;
    }
    else
    {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        p = atoi(argv[3]);
        nwu = atoi(argv[4]);
        nrep = atoi(argv[5]);
    }

    gpuXmm_time_t tdiff[NB_META];

    unsigned int size_a = m * n * sizeof(gpuXmm_precision_t);
    unsigned int size_b = n * p * sizeof(gpuXmm_precision_t);
    unsigned int size_c = m * p * sizeof(gpuXmm_precision_t);
    
    gpuXmm_precision_t *a = (gpuXmm_precision_t*)malloc(size_a);
    gpuXmm_precision_t *b = (gpuXmm_precision_t*)malloc(size_b);

    srand(0);
    random_Xarray_2D(m, n, a);
    random_Xarray_2D(n, p, b);

    #ifdef __GPUXMM_NEED_HANDLE
    gpuXmm_handle_t handle;
    gpuXmm_handle_create(handle);
    #endif

    for (unsigned int i_meta = 0; i_meta < NB_META; i_meta++)
    {
        gpuXmm_precision_t *c = (gpuXmm_precision_t*)malloc(size_c);

        if ( i_meta == 0 )
        {
            for (unsigned int i = 0; i < nwu; i++)
            {
                #ifdef __GPUXMM_NEED_HANDLE
                kernel_gpuXmm(handle, m, n, p, a, b, c);
                #else
                kernel_gpuXmm(m, n, p, a, b, c);
                #endif
            }
        }
        else
        {
            #ifdef __GPUXMM_NEED_HANDLE
            kernel_gpuXmm(handle, m, n, p, a, b, c);
            #else
            kernel_gpuXmm(m, n, p, a, b, c);
            #endif
        }

        const gpuXmm_time_t t1 = measure_clock();
        for (unsigned int i = 0; i < nrep; i++)
        {
            #ifdef __GPUXMM_NEED_HANDLE
            kernel_gpuXmm(handle, m, n, p, a, b, c);
            #else
            kernel_gpuXmm(m, n, p, a, b, c);
            #endif
        }
        const gpuXmm_time_t t2 = measure_clock();

        tdiff[i_meta] = t2 - t1;
        
        free(c);
    }

    #ifdef __GPUXMM_NEED_HANDLE
    gpuXmm_handle_destroy(handle);
    #endif    

    free(a);
    free(b);

    print_measure(m, n, p, nrep, tdiff);
    
    return EXIT_SUCCESS;
}