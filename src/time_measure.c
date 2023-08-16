#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include "time_measure.h"

#ifdef __GPUXMM_RDTSC_ENABLE
extern uint64_t rdtsc ();
uint64_t measure_clock()
{
    return rdtsc();
}
#else
#include <omp.h>
double measure_clock()
{
    return omp_get_wtime();    
}
#endif