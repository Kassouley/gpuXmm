
GPU = AMD
PRECISION = DP

IS_KERNEL_CPU 		:= $(filter $(KERNEL), BASIS CPU_OMP CBLAS ARMPL MKL)
IS_KERNEL_OMP 		:= $(filter $(KERNEL), GPU_OMP GPU_OMP_WO_DT)
IS_KERNEL_ACC 		:= $(filter $(KERNEL), ACC ACC_WO_DT)
IS_KERNEL_HIP 		:= $(filter $(KERNEL), HIP HIP_WO_DT)
IS_KERNEL_ROCBLAS 	:= $(filter $(KERNEL), ROCBLAS ROCBLAS_WO_DT)
IS_KERNEL_CUDA 		:= $(filter $(KERNEL), CUDA CUDA_WO_DT)
IS_KERNEL_CUBLAS 	:= $(filter $(KERNEL), CUBLAS CUBLAS_WO_DT)

# -------------------- CC -------------------- #

ifneq ($(filter $(KERNEL), HIP HIP_WO_DT ROCBLAS ROCBLAS_WO_DT),)
	CC=hipcc
else ifneq ($(filter $(KERNEL), CUDA CUDA_WO_DT CUBLAS CUBLAS_WO_DT),)
	CC=nvcc
	OMP_FLAG = -Xcompiler -fopenmp
else ifeq ($(KERNEL), ARMPL)
	CC=armclang
else
	ifeq ($(GPU), AMD)
		CC=/opt/rocm/llvm/bin/clang
	else ifeq ($(GPU), NVIDIA)
		CC=gcc
	endif
endif
OMP_FLAG ?= -fopenmp

# ---------------- INCLUDE DIR---------------- #

INC_DIRS := $(shell find ./include -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

# ----------------- CPP FLAGS----------------- #

CPPFLAGS=$(INC_FLAGS) -D $(KERNEL) -D $(PRECISION)

ifeq ($(METRIC), RDTSC-Cycles)
	CPPFLAGS += -D __GPUXMM_RDTSC_ENABLE
endif

# ------------------ CFLAGS ------------------ #

CFLAGS = -g -O3 

# ------------------ LFLAGS ------------------ #

LFLAGS = $(OMP_FLAG) -lm
ifeq ($(KERNEL),CBLAS)
	LFLAGS += -lblas
else ifeq ($(KERNEL),MKL)
	LFLAGS += -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lm
else ifneq ($(IS_KERNEL_ROCBLAS),)
	LFLAGS += -lrocblas  -L/opt/rocm-5.4.3/rocblas/lib/librocblas.so  -I/opt/rocm-5.4.3/include/
else ifneq ($(IS_KERNEL_CUBLAS),)
	LFLAGS += -lcublas
endif

# ----------------- OPT_FLAGS ----------------- #

ifeq ($(KERNEL),CPU_OMP)
	OPT_FLAGS=-fopenmp
else ifeq ($(KERNEL),ARMPL)
	OPT_FLAGS=-armpl
else ifneq ($(IS_KERNEL_OMP),)
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
else ifneq ($(filter $(KERNEL), ACC ACC_WO_DT),)
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

SRC_DIR=./src
KERNEL_DIR=$(SRC_DIR)/kernel
BENCH_DIR=$(SRC_DIR)/bench
CHECK_DIR=$(SRC_DIR)/check

IS_KERNEL_IN_C 			:= $(filter $(KERNEL), BASIS CPU_OMP CBLAS ARMPL MKL GPU_OMP ACC)
IS_KERNEL_IN_C_WO_DT 	:= $(filter $(KERNEL), GPU_OMP_WO_DT ACC_WO_DT)
IS_KERNEL_IN_CPP 		:= $(filter $(KERNEL), HIP ROCBLAS CUDA CUBLAS)
IS_KERNEL_IN_CPP_WO_DT 	:= $(filter $(KERNEL), HIP_WO_DT ROCBLAS_WO_DT CUDA_WO_DT CUBLAS_WO_DT)

ifneq ($(IS_KERNEL_IN_C),)
	SRC_CHECKER=checker.c
else ifneq ($(IS_KERNEL_IN_C_WO_DT),)
	SRC_CHECKER=checker_wo_dt.c
else ifneq ($(IS_KERNEL_IN_CPP),)
	SRC_CHECKER=checker.cpp
else ifneq ($(IS_KERNEL_IN_CPP_WO_DT),)
	SRC_CHECKER=checker_wo_dt.cpp
endif

ifneq ($(IS_KERNEL_IN_C),)
	SRC_DRIVER=driver.c
else ifneq ($(IS_KERNEL_IN_C_WO_DT),)
	SRC_DRIVER=driver_wo_dt.c
else ifneq ($(IS_KERNEL_IN_CPP),)
	SRC_DRIVER=driver.cpp
else ifneq ($(IS_KERNEL_IN_CPP_WO_DT),)
	SRC_DRIVER=driver_wo_dt.cpp
endif

ifneq ($(IS_KERNEL_CPU),)
	SRC_KERNEL=kernel_cpu.c 
else ifneq ($(IS_KERNEL_OMP),)
	SRC_KERNEL=kernel_device_omp.c
else ifneq ($(IS_KERNEL_ACC),)
	SRC_KERNEL=/kernel_device_acc.c
else ifneq ($(IS_KERNEL_HIP),)
	SRC_KERNEL=kernel_device_hip.cpp
else ifneq ($(IS_KERNEL_ROCBLAS),)
	SRC_KERNEL=kernel_device_rocblas.cpp
else ifneq ($(IS_KERNEL_CUDA),)
	SRC_KERNEL=kernel_device_cuda.cu
else ifneq ($(IS_KERNEL_CUBLAS),)
	SRC_KERNEL=kernel_device_cublas.cu
endif


BUILD_DIR=./build
OBJS_COMMON=$(BUILD_DIR)/array.o $(BUILD_DIR)/kernel.o
OBJS_MEASURE=$(OBJS_COMMON) $(BUILD_DIR)/print_measure.o $(BUILD_DIR)/time_measure.o

ifeq ($(METRIC), RDTSC-Cycles)
	OBJS_MEASURE += $(BUILD_DIR)/rdtsc.o 
endif

all: check measure


.PHONY: $(BUILD_DIR)/kernel.o $(BUILD_DIR)/driver.o $(BUILD_DIR)/driver_check.o

$(BUILD_DIR)/driver_check.o: $(CHECK_DIR)/$(SRC_CHECKER)
	@echo Building $@ \($(KERNEL)\) . . .
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(OPT_FLAGS) -c $< -o $@

$(BUILD_DIR)/driver.o: $(BENCH_DIR)/$(SRC_DRIVER)
	@echo Building $@ \($(KERNEL)\) . . .
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(OPT_FLAGS) -c $< -o $@

$(BUILD_DIR)/kernel.o: $(KERNEL_DIR)/$(SRC_KERNEL)
	@echo Building $@ \($(KERNEL)\) . . .
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(OPT_FLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@echo Building $@ . . .
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

check: $(OBJS_COMMON) $(BUILD_DIR)/driver_check.o
	@echo Building $@ . . .
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS) $(CPPFLAGS) $(OPT_FLAGS)


measure: $(OBJS_MEASURE) $(BUILD_DIR)/driver.o
	@echo Building $@ . . .
	$(CC) -o $@ $^ $(CFLAGS) $(CMEASURE) $(CPPFLAGS) $(LFLAGS) $(OPT_FLAGS)

clean:
	rm -rf ./build check calibrate measure

