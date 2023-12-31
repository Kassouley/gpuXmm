#include <stdio.h>
#include "gpuXmm.h"
#include "kernel.h"

#ifdef BASIS
void kernel_gpuXmm (unsigned int m, unsigned int n, unsigned int p, 
                    const gpuXmm_precision_t* a, const gpuXmm_precision_t* b, gpuXmm_precision_t* c)
{
    for(unsigned int i = 0; i < m; i++)
    {
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
#endif

#ifdef CPU_OMP
#include <omp.h>
void kernel_gpuXmm (unsigned int m, unsigned int n, unsigned int p, 
                    const gpuXmm_precision_t* a, const gpuXmm_precision_t* b, gpuXmm_precision_t* c)
{
    #pragma omp parallel for schedule(dynamic)
    for(unsigned int i = 0; i < m; i++)
    {
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
#endif

#ifdef CBLAS
#include <cblas.h>
void kernel_gpuXmm (unsigned int m, unsigned int n, unsigned int p, 
                    const gpuXmm_precision_t* a, const gpuXmm_precision_t* b, gpuXmm_precision_t* c)
{
    #ifdef SP
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                m, p, n, 1.0, a, n, b, p, 0.0, c, p);
    #else //DP
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                m, p, n, 1.0, a, n, b, p, 0.0, c, p);
    #endif
}
#endif

#ifdef ARMPL
#include <armpl.h>
void kernel_gpuXmm (unsigned int m, unsigned int n, unsigned int p, 
                    const gpuXmm_precision_t* a, const gpuXmm_precision_t* b, gpuXmm_precision_t* c)
{
    gpuXmm_precision_t alpha = 1.0, beta = 0.0;
    armpl_int_t mm = (armpl_int_t) m;
    armpl_int_t nn = (armpl_int_t) n;
    armpl_int_t pp = (armpl_int_t) p;
    #ifdef SP
    sgemm_("N", "N", &pp, &mm, &nn, &alpha, b, &pp, a, &nn, &beta, c, &pp);
    #else //DP
    dgemm_("N", "N", &pp, &mm, &nn, &alpha, b, &pp, a, &nn, &beta, c, &pp);
    #endif
}
#endif

#ifdef MKL
#include <mkl.h>
void kernel_gpuXmm (unsigned int m, unsigned int n, unsigned int p, 
                    const gpuXmm_precision_t* a, const gpuXmm_precision_t* b, gpuXmm_precision_t* c)
{
    #ifdef SP
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, p, n, 1.0, a, n, b, p, 0.0, c, p);    
    #else //DP
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, p, n, 1.0, a, n, b, p, 0.0, c, p);
    #endif
}
#endif
