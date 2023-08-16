#ifndef __ARRAY_C
#define __ARRAY_C

void random_Darray_2D(unsigned int row, unsigned int col, double* array);
void random_Sarray_2D(unsigned int row, unsigned int col, float* array);
void print_Darray_2D(unsigned int row, unsigned int col, double* array);
void print_Sarray_2D(unsigned int row, unsigned int col, float* array);

#ifdef SP
    #define random_Xarray_2D(row, col, array) \
    {\
        random_Sarray_2D(row, col, array); \
    }
    #define print_Xarray_2D(row, col, array) \
    {\
        print_Sarray_2D(row, col, array); \
    }
#else //DP
    #define random_Xarray_2D(row, col, array) \
    {\
        random_Darray_2D(row, col, array); \
    }
    #define print_Xarray_2D(row, col, array) \
    {\
        print_Darray_2D(row, col, array); \
    }
#endif


#endif