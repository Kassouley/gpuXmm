#include <stdio.h>
#include <openacc.h>
#include "gpuXmm.h"
#include "kernel.h"

#ifdef OPENACC
void kernel_gpuXmm (unsigned int m, unsigned int n, unsigned int p, 
                    const gpuXmm_precision_t* a, const gpuXmm_precision_t* b, gpuXmm_precision_t* c)
{
    #pragma acc data copyin(a[0:m*n], b[0:n*p]) copyout(c[0:m*p])
    {
        # pragma acc region
        {
            # pragma acc loop independent vector(32) 
            for(unsigned int i = 0; i < m; i++)
            {
                # pragma acc loop independent vector(32) 
                for(unsigned int j = 0; j < p; j++)
                {
                    c[i*p+j] = 0;
                    for(unsigned int k = 0; k < n; k++)
                    {
                        c[i*p+j] += a[i*n+k] * b[k*p+j];
                    }
                }
            }
        }
    }
}
#endif

#ifdef OPENACC_WO_DT
void kernel_gpuXmm (unsigned int m, unsigned int n, unsigned int p, 
                    const gpuXmm_precision_t* a, const gpuXmm_precision_t* b, gpuXmm_precision_t* c)
{
    #pragma acc data deviceptr(a, b, c)
    {
        #pragma acc region
        {
            #pragma acc loop independent vector(32) 
            for(unsigned int i = 0; i < m; i++)
            {
                # pragma acc loop independent vector(32) 
                for(unsigned int j = 0; j < p; j++)
                {
                    c[i*p+j] = 0;
                    for(unsigned int k = 0; k < n; k++)
                    {
                        c[i*p+j] += a[i*n+k] * b[k*p+j];
                    }
                }
            }
        }
    }
}
#endif