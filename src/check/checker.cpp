#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpuXmm.h"
#include "kernel.h"
extern "C" {
#include "array.h"
}

#define OUTOUT_FILE "output_check.txt"

int main(int argc, char **argv)
{
    unsigned int m = 0, n = 0, p = 0;
    char* file_name = NULL;
    FILE * output = NULL;

    if (argc != 4 && argc != 5) 
    {
        fprintf (stderr, "Usage: %s <m> <n> <p> [file name]\n", argv[0]);
        return 1;
    }
    else
    {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        p = atoi(argv[3]);
        file_name = (char*)malloc(256*sizeof(char));
        if (argc == 4)
            strcpy(file_name, OUTOUT_FILE);
        else if (argc == 5)
            strcpy(file_name, argv[4]);
    }
    
    int size_a = m * n * sizeof(gpuXmm_precision_t);
    int size_b = n * p * sizeof(gpuXmm_precision_t);
    int size_c = m * p * sizeof(gpuXmm_precision_t);
    
    gpuXmm_precision_t *a = (gpuXmm_precision_t*)malloc(size_a);
    gpuXmm_precision_t *b = (gpuXmm_precision_t*)malloc(size_b);
    gpuXmm_precision_t *c = (gpuXmm_precision_t*)malloc(size_c);

    srand(0);
    random_Xarray_2D(m, n, a);
    random_Xarray_2D(n, p, b);

    #ifdef __GPUXMM_NEED_HANDLE
    gpuXmm_handle_t handle;
    gpuXmm_handle_create(handle);

    kernel_gpuXmm(handle, m, n, p, a, b, c);

    gpuXmm_handle_destroy(handle);
    #else
    kernel_gpuXmm(m, n, p, a, b, c);
    #endif

    output = fopen(file_name, "w");
    for (unsigned int i = 0; i < m; i++)
    {
        for (unsigned int j = 0; j < p; j++)
        {
            fprintf(output, "%f ", c[i*p+j]);
        }
        fprintf(output, "\n");
    }
    fclose(output);
    
    free(file_name);
    free(a);
    free(b);
    free(c);
}
