#ifndef PRINT_CALIB_H
#define PRINT_CALIB_H
#ifndef NB_META
#define NB_META 31
#endif
#include <stdint.h>
void print_calib(unsigned int repm, uint64_t** tdiff, char* file_name);
static int cmp_uint64 (const void *a, const void *b);
#endif