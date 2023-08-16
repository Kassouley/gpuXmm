#ifndef PRINT_MEASURE_H
#define PRINT_MEASURE_H
#ifndef NB_META
#define NB_META 31
#endif
#include <stdint.h>

#ifdef __GPUXMM_RDTSC_ENABLE
void print_measure(unsigned int m, unsigned int n, unsigned int p, 
                            unsigned int nrep, uint64_t tdiff[NB_META]);
#else
void print_measure(unsigned int m, unsigned int n, unsigned int p, 
                        unsigned int nrep, double tdiff[NB_META]);
#endif                            
                            
static int cmp_uint64 (const void *a, const void *b);
#endif