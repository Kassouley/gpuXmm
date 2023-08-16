
GPU ?= AMD
PRECISION ?= DP
METRIC_ENABLE=__GPUXMM_RDTSC_DISABLE
ifeq ($(METRIC), RDTSC-Cycles)
	METRIC_ENABLE=__GPUXMM_RDTSC_ENABLE
endif

# -------------------- CC -------------------- #
ifneq ($(filter $(KERNEL), HIP HIP_WO_DT ROCBLAS ROCBLAS_WO_DT),)
	CC?=hipcc
else ifneq ($(filter $(KERNEL), CUDA CUDA_WO_DT CUBLAS CUBLAS_WO_DT),)
	CC?=nvcc
else
	ifeq ($(GPU), AMD)
		CC?=/opt/rocm/llvm/bin/clang
	else ifeq ($(GPU), NVIDIA)
		CC?=nvc
	endif
endif

# ------------------ CFLAGS ------------------ #

CFLAGS=-g -O3 -lm -I./include -I./include/gpuXmm -D $(KERNEL) -D $(PRECISION) -fopenmp
CMEASURE=-D $(METRIC_ENABLE)

# ------------------ LFLAGS ------------------ #

ifeq ($(KERNEL),CBLAS)
	LFLAGS=-lblas
else ifneq ($(filter $(KERNEL), ROCBLAS ROCBLAS_WO_DT),)
	LFLAGS=-lrocblas  -L/opt/rocm-5.4.3/rocblas/lib/librocblas.so  -I/opt/rocm-5.4.3/include/
else ifneq ($(filter $(KERNEL), CUBLAS CUBLAS_WO_DT),)
	LFLAGS=-lcublas
endif

# ----------------- OPT_FLAGS ----------------- #

ifeq ($(KERNEL),CPU_OMP)
	OPT_FLAGS=-fopenmp
else ifneq ($(filter $(KERNEL), GPU_OMP GPU_OMP_WO_DT),)
	ifeq ($(GPU), AMD)
		OPT_FLAGS=-fopenmp=libomp -target x86_64-pc-linux-gnu \
		-fopenmp-targets=amdgcn-amd-amdhsa \
		-Xopenmp-target=amdgcn-amd-amdhsa -march=gfx1030
	else ifeq ($(GPU), NVIDIA)
		ifeq ($(CC),gcc)
			OPT_FLAGS=-fopenmp
		else
			OPT_FLAGS=-fopenmp -mp=gpu -Minfo=mp
		endif
	endif
else ifneq ($(filter $(KERNEL), OPENACC OPENACC_WO_DT),)
	ifeq ($(CC),gcc)
		OPT_FLAGS=-fopenacc
	else
		OPT_FLAGS=-acc -Minfo=accel 
	endif
else ifneq ($(filter $(KERNEL), CUDA CUDA_WO_DT CUBLAS CUBLAS_WO_DT),)
	OPT_FLAGS=-gencode=arch=compute_52,code=sm_52 \
  			  -gencode=arch=compute_60,code=sm_60 \
  			  -gencode=arch=compute_61,code=sm_61 \
  			  -gencode=arch=compute_70,code=sm_70 \
  			  -gencode=arch=compute_75,code=sm_75 \
  			  -gencode=arch=compute_80,code=sm_80 \
  			  -gencode=arch=compute_80,code=compute_80
endif

# ------------------- SRC ------------------- #

SRC_COMMON=src/array.c 

SRC_DIR=./src
KERNEL_DIR=$(SRC_DIR)/kernel
BENCH_DIR=$(SRC_DIR)/bench
CHECK_DIR=$(SRC_DIR)/check
CALIB_DIR=$(SRC_DIR)/calibrate

IS_KERNEL_IN_C := $(filter $(KERNEL), BASIS CPU_OMP CBLAS GPU_OMP OPENACC)
IS_KERNEL_IN_C_WO_DT := $(filter $(KERNEL), GPU_OMP_WO_DT OPENACC_WO_DT)
IS_KERNEL_IN_CPP := $(filter $(KERNEL), HIP ROCBLAS CUDA CUBLAS)
IS_KERNEL_IN_CPP_WO_DT := $(filter $(KERNEL), HIP_WO_DT ROCBLAS_WO_DT CUDA_WO_DT CUBLAS_WO_DT)

ifneq ($(IS_KERNEL_IN_C),)
	SRC_CHECKER=$(CHECK_DIR)/checker.c
else ifneq ($(IS_KERNEL_IN_C_WO_DT),)
	SRC_CHECKER=$(CHECK_DIR)/checker_wo_dt.c
else ifneq ($(IS_KERNEL_IN_CPP),)
	SRC_CHECKER=$(CHECK_DIR)/checker.cpp
else ifneq ($(IS_KERNEL_IN_CPP_WO_DT),)
	SRC_CHECKER=$(CHECK_DIR)/checker_wo_dt.cpp
endif

ifneq ($(IS_KERNEL_IN_C),)
	SRC_DRIVER=$(BENCH_DIR)/driver.c
else ifneq ($(IS_KERNEL_IN_C_WO_DT),)
	SRC_DRIVER=$(BENCH_DIR)/driver_wo_dt.c
else ifneq ($(IS_KERNEL_IN_CPP),)
	SRC_DRIVER=$(BENCH_DIR)/driver.cpp
else ifneq ($(IS_KERNEL_IN_CPP_WO_DT),)
	SRC_DRIVER=$(BENCH_DIR)/driver_wo_dt.cpp
endif

IS_KERNEL_CPU := $(filter $(KERNEL), BASIS CPU_OMP CBLAS)
IS_KERNEL_OMP := $(filter $(KERNEL), GPU_OMP GPU_OMP_WO_DT)
IS_KERNEL_ACC := $(filter $(KERNEL), OPENACC OPENACC_WO_DT)
IS_KERNEL_HIP := $(filter $(KERNEL), HIP HIP_WO_DT)
IS_KERNEL_ROCBLAS := $(filter $(KERNEL), ROCBLAS ROCBLAS_WO_DT)
IS_KERNEL_CUDA := $(filter $(KERNEL), CUDA CUDA_WO_DT)
IS_KERNEL_CUBLAS := $(filter $(KERNEL), CUBLAS CUBLAS_WO_DT)

ifneq ($(IS_KERNEL_CPU),)
	SRC_KERNEL=$(KERNEL_DIR)/kernel_cpu.c 
else ifneq ($(IS_KERNEL_OMP),)
	SRC_KERNEL=$(KERNEL_DIR)/kernel_device_omp.c
else ifneq ($(IS_KERNEL_ACC),)
	SRC_KERNEL=$(KERNEL_DIR)/kernel_device_acc.c
else ifneq ($(IS_KERNEL_HIP),)
	SRC_KERNEL=$(KERNEL_DIR)/kernel_device_hip.cpp
else ifneq ($(IS_KERNEL_ROCBLAS),)
	SRC_KERNEL=$(KERNEL_DIR)/kernel_device_rocblas.cpp
else ifneq ($(IS_KERNEL_CUDA),)
	SRC_KERNEL=$(KERNEL_DIR)/kernel_device_cuda.cu
else ifneq ($(IS_KERNEL_CUBLAS),)
	SRC_KERNEL=$(KERNEL_DIR)/kernel_device_cublas.cu
endif

BUILD_DIR=./build
OBJS_COMMON=$(BUILD_DIR)/array.o $(BUILD_DIR)/kernel.o
OBJS_MEASURE=

ifeq ($(METRIC), RDTSC-Cycles)
	OBJS_COMMON += $(OBJS_DIR)/rdtsc.o 
endif

all: check measure

check: $(SRC_COMMON)
	$(CC) -o $@ $^ $(SRC_KERNEL) $(SRC_CHECKER) $(CFLAGS) $(LFLAGS) $(OPT_FLAGS)

measure: $(SRC_COMMON) src/rdtsc.c src/print_measure.c src/time_measure.c 
	$(CC) -o $@ $^ $(SRC_KERNEL) $(SRC_DRIVER) $(CFLAGS) $(CMEASURE) $(LFLAGS) $(OPT_FLAGS)

# OBJS_COMMON=tab.o rdtsc.o kernel.o

# all: check calibrate measure

# check: $(OBJS_COMMON) driver_check.o
# 	$(CC) -o $@ $^ -lm -fopenmp
# calibrate: $(OBJS_COMMON) driver_calib.o
# 	$(CC) -o $@ $^ -lm -fopenmp
# measure: $(OBJS_COMMON) driver.o
# 	$(CC) -o $@ $^ -lm -fopenmp

# driver_check.o: driver_check.c
# 	$(CC) $(CFLAGS) -c $< -o $@
# driver_calib.o: driver_calib.c
# 	$(CC) $(CFLAGS) -c $< -o $@
# driver.o: driver.c
# 	$(CC) $(CFLAGS) -c $< -o $@

# kernel.o: kernel.c
# 	$(CC) $(OPTFLAGS) -D $(OPT) -fopenmp -c $< -o $@

clean:
	rm -rf *.o check calibrate measure

