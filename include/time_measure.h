#ifndef MEASURE_TIME_H
#define MEASURE_TIME_H

#ifdef __GPUXMM_RDTSC_ENABLE
#include <stdint.h>
typedef uint64_t gpuXmm_time_t;
#else
typedef double gpuXmm_time_t;
#endif
gpuXmm_time_t measure_clock();
#endif