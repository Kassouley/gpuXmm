#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include "time_measure.h"
#include "print_measure.h"
#ifndef NB_META
#define NB_META 31
#endif
#define OUTPUT_FILE "output/tmp/measure_tmp.out"

static int cmp_uint64 (const void *a, const void *b)
{
    const uint64_t va = *((uint64_t *) a);
    const uint64_t vb = *((uint64_t *) b);

    if (va < vb) return -1;
    if (va > vb) return 1;
    return 0;
}

#ifdef __GPUXMM_RDTSC_ENABLE
void print_measure(unsigned int m, unsigned int n, unsigned int p, 
                            unsigned int nrep, uint64_t tdiff[NB_META])
{
    FILE * output = NULL;
    
    const unsigned long nbitr = (unsigned long)m*(unsigned long)n*(unsigned long)p*(unsigned long)nrep;

    qsort (tdiff, NB_META, sizeof tdiff[0], cmp_uint64);
    printf("Minimum : %.6g RDTSC-Cycles (%.3g par itération)\n", (float)tdiff[0]/(float)nrep        , (float)tdiff[0]/(float)nbitr);
    printf("Median  : %.6g RDTSC-Cycles (%.3g par itération)\n", (float)tdiff[NB_META/2]/(float)nrep, (float)tdiff[NB_META/2]/(float)nbitr);
    printf("Maximum : %.6g RDTSC-Cycles (%.3g par itération)\n", (float)tdiff[NB_META-1]/(float)nrep, (float)tdiff[NB_META-1]/(float)nbitr);
    
    const float stabilite = (tdiff[NB_META/2] - tdiff[0]) * 100.0f / tdiff[0];
    
    if (stabilite >= 10)
        printf("Bad Stability : %.2f %%\n", stabilite);
    else if ( stabilite >= 5 )
        printf("Average Stability : %.2f %%\n", stabilite);
    else
        printf("Good Stability : %.2f %%\n", stabilite);

    output = fopen(OUTPUT_FILE, "a");
    if (output != NULL) 
    {
        fprintf(output, " %5d , %5d , %5d , %14f , %14f , %14f , %10f\n", 
                m, n, p,
                (float)tdiff[0]/nrep, 
                (float)tdiff[NB_META/2]/nrep, 
                (float)tdiff[NB_META/2]/nbitr, 
                stabilite);
        fclose(output);
    }
    else
    {
        char cwd[1028];
        if (getcwd(cwd, sizeof(cwd)) != NULL) 
        {
            printf("Couldn't open '%s/%s' file\n Measure not saved\n", cwd, OUTPUT_FILE);
        }
    }
}
#else
void print_measure(unsigned int m, unsigned int n, unsigned int p, 
                        unsigned int nrep, double tdiff[NB_META])
{
    FILE * output = NULL;

    qsort (tdiff, NB_META, sizeof tdiff[0], cmp_uint64);

    const double nb_gflops = (double)((m*p)*(2*n-1)) * 1e-9;
    const double time_min  = (double)tdiff[0]/(double)nrep;
    const double time_med  = (double)tdiff[NB_META/2]/(double)nrep;
    const float stabilite  = (tdiff[NB_META/2] - tdiff[0]) * 100.0f / tdiff[0];

    double rate = 0.0, drate = 0.0;
    for (unsigned int i = 0; i < NB_META; i++)
    {
        rate += nb_gflops / (tdiff[i]/nrep);
        drate += (nb_gflops * nb_gflops) / ((tdiff[i]/nrep) * (tdiff[i]/nrep));
    }
    rate /= (double)(NB_META);
    drate = sqrt(drate / (double)(NB_META) - (rate * rate));
  
    printf("-----------------------------------------------------\n");

    printf("Time (minimum, ms): %13s %10.5f ms\n", "", time_min * 1e3);
    printf("Time (median, ms):  %13s %10.5f ms\n", "", time_med * 1e3);
    
    if (stabilite >= 10)
        printf("Bad Stability: %18s %10.2f %%\n", "", stabilite);
    else if ( stabilite >= 5 )
        printf("Average Stability: %14s %10.2f %%\n", "", stabilite);
    else
        printf("Good Stability: %17s %10.2f %%\n", "", stabilite);

    printf("\033[1m%s %4s \033[42m%10.2lf +- %.2lf GFLOP/s\033[0m\n",
        "Average performance:", "", rate, drate);
    printf("-----------------------------------------------------\n");
    

    output = fopen(OUTPUT_FILE, "a");
    if (output != NULL) 
    {
        
        fprintf(output, " %5d , %5d , %5d , %14f , %14f , %14f , %10f\n", 
                m, n, p,
                rate, 
                time_min*1e3, 
                time_med*1e3,
                stabilite);
        fclose(output);
    }
    else
    {
        char cwd[1028];
        if (getcwd(cwd, sizeof(cwd)) != NULL) 
        {
            printf("Couldn't open '%s/%s' file\n Measure not saved\n", cwd, OUTPUT_FILE);
        }
    }
}
#endif