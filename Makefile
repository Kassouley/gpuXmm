#-----------------------------------------------------------------------


# ---------------- ENV VAR ---------------- #
CC			?= gcc
CFLAGS 		?= -g -O3
KERNEL		?= BASIS
PRECISION   ?= SP
USETX		?= NO
METRIC		?= GFLOPS

KERNEL_CPU         = kernel_cpu.c 
KERNEL_GPU_OMP     = kernel_device_omp.c
KERNEL_GPU_ACC     = kernel_device_acc.c
KERNEL_GPU_HIP     = kernel_device_hip.cpp
KERNEL_GPU_ROCBLAS = kernel_device_rocblas.cpp
KERNEL_GPU_CUDA    = kernel_device_cuda.cu
KERNEL_GPU_CUBLAS  = kernel_device_cublas.cu
SRC_KERNEL 		   = $(KERNEL_CPU)

KERNEL_API		   = $(subst _WO_DT,,$(KERNEL))

# ---------------- DIR PATH ---------------- #
SRC_DIR	     = ./src
KERNEL_DIR   = $(SRC_DIR)/kernel
BENCH_DIR    = $(SRC_DIR)/bench
CHECK_DIR    = $(SRC_DIR)/check

BUILD_DIR    = ./build
OBJS_COMMON  = $(BUILD_DIR)/array.o $(BUILD_DIR)/kernel.o
OBJS_MEASURE = $(OBJS_COMMON) $(BUILD_DIR)/print_measure.o $(BUILD_DIR)/time_measure.o
ifeq ($(METRIC), RDTSC-Cycles)
	OBJS_MEASURE += $(BUILD_DIR)/rdtsc.o 
endif

# ---------------- DRIVER FILES ---------------- #
IS_KERNEL_IN_C 			:= $(filter $(KERNEL), BASIS CPU_OMP CBLAS ARMPL MKL GPU_OMP ACC)
IS_KERNEL_IN_C_WO_DT 	:= $(filter $(KERNEL), GPU_OMP_WO_DT ACC_WO_DT)
IS_KERNEL_IN_CPP 		:= $(filter $(KERNEL), HIP ROCBLAS CUDA CUBLAS)
IS_KERNEL_IN_CPP_WO_DT 	:= $(filter $(KERNEL), HIP_WO_DT ROCBLAS_WO_DT CUDA_WO_DT CUBLAS_WO_DT)

ifneq ($(IS_KERNEL_IN_C),)
	SRC_DRIVER  = driver.c
	SRC_CHECKER = checker.c
else ifneq ($(IS_KERNEL_IN_C_WO_DT),)
	SRC_DRIVER  = driver_wo_dt.c
	SRC_CHECKER = checker_wo_dt.c
else ifneq ($(IS_KERNEL_IN_CPP),)
	SRC_DRIVER  = driver.cpp
	SRC_CHECKER = checker.cpp
else ifneq ($(IS_KERNEL_IN_CPP_WO_DT),)
	SRC_DRIVER  = driver_wo_dt.cpp
	SRC_CHECKER = checker_wo_dt.cpp
endif

# ---------------- INCLUDE DIR ---------------- #
INC_DIRS   := $(shell find ./include -type d)
INC_FLAGS  := $(addprefix -I,$(INC_DIRS))

CFLAGS += $(INC_FLAGS) 

# ---------------- DFLAGS ---------------- #
DFLAGS		= -D $(KERNEL) -D $(PRECISION) -D $(USETX) 
ifeq ($(METRIC), RDTSC-Cycles)
	DFLAGS += -D __GPUXMM_RDTSC_ENABLE
else
	CFLAGS += -fopenmp
endif

CFLAGS += $(DFLAGS)

# ------------------ LFLAGS ------------------ #
LFLAGS  = -lm

# ------------------ CPU & GPU TARGET ------------------ #
UNAMEP    = $(shell uname -p)
CPUTARGET = $(UNAMEP)-pc-linux-gnu
ifeq ($(UNAMEP),ppc64le)
	CPUTARGET = ppc64le-linux-gnu
endif

HAS_ROCM := $(shell command -v rocm-smi 2> /dev/null)
HAS_ROCM := $(shell command -v nvcc --version 2> /dev/null)

# ---------------- CPU COMP & FLAGS ---------------- #
ifeq ($(KERNEL_API), CPU_OMP)
	CFLAGS 		   += -fopenmp
	SRC_KERNEL 		= $(KERNEL_CPU)
else ifeq ($(KERNEL_API), CBLAS)
	LFLAGS 		   += -lblas
	SRC_KERNEL 		= $(KERNEL_CPU)
else ifeq ($(KERNEL_API), MKL)
	LFLAGS 		   += -lmkl_rt
	SRC_KERNEL 		= $(KERNEL_CPU)
else ifeq ($(KERNEL_API), ARMPL)
	CC				= armclang		
	LFLAGS 		   += -larmpl_mp
	SRC_KERNEL 		= $(KERNEL_CPU)
		
# ---------------- ROCM COMP & FLAGS ---------------- #
else ifneq ($(HAS_ROCM),)
	ROCM_PATH 	   ?= /opt/rocm/
	ROCM_GPUTARGET ?= amdgcn-amd-amdhsa

	INSTALLED_GPU   = $(shell $(ROCM_PATH)/bin/offload_arch | grep -m 1 -E gfx[^0]{1})
	ROCM_GPU       ?= $(INSTALLED_GPU)

	ifeq ($(KERNEL_API),GPU_OMP)
		CC          = $(ROCM_PATH)/llvm/bin/clang
		CFLAGS 	   += -target $(CPUTARGET) -fopenmp -fopenmp-targets=$(ROCM_GPUTARGET) -Xopenmp-target=$(ROCM_GPUTARGET) -march=$(ROCM_GPU)
		SRC_KERNEL 	= $(KERNEL_GPU_OMP)
	else ifeq ($(KERNEL_API),HIP)
		CC			= hipcc
		SRC_KERNEL  = $(KERNEL_GPU_HIP)
	else ifeq ($(KERNEL_API),ROCBLAS)
		CC			= hipcc
		LFLAGS 	   += -lrocblas -L$(ROCM_PATH)/rocblas/lib/librocblas.so -I$(ROCM_PATH)/include/
		SRC_KERNEL  = $(KERNEL_GPU_ROCBLAS)
	endif
	ifeq ($(USETX),YES)
		LFLAGS	   += -lroctx64 -lroctracer64 
	endif

# ---------------- CUDA COMP & FLAGS ---------------- #
else ifneq ($(HAS_CUDA),)
	ifeq ($(KERNEL_API),GPU_OMP)
		CC          = nvc
		CFLAGS     += -fopenmp -mp=gpu -Minfo=mp
		SRC_KERNEL  = $(KERNEL_GPU_OMP)
	else ifeq ($(KERNEL_API),ACC)
		ifeq ($(CC),gcc)
			CFLAGS += -fopenaccc
		else
			CC 		= nvc
			CFLAGS += -acc -Minfo=accel 
		endif
		SRC_KERNEL  = $(KERNEL_GPU_ACC)
	else ifeq ($(KERNEL_API),CUDA)
		CC			= nvcc
		CFLAGS     += -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=compute_80
		SRC_KERNEL  = $(KERNEL_GPU_CUDA)
	else ifeq ($(KERNEL_API),CUBLAS)
		CC			= nvcc
		CFLAGS     += -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=compute_80
		LFLAGS 	   += -lcublas
		SRC_KERNEL  = $(KERNEL_GPU_CUBLAS)
	endif
	ifeq ($(USETX),YES)
		LFLAGS 	   += -lnvToolsExt -ldl
	endif
endif

# ---------------- BUILD ---------------- #
all: check measure

.PHONY: $(BUILD_DIR)/kernel.o $(BUILD_DIR)/driver.o $(BUILD_DIR)/driver_check.o

$(BUILD_DIR)/driver_check.o: $(CHECK_DIR)/$(SRC_CHECKER)
	@echo Building $@ \($(KERNEL)\) . . .
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/driver.o: $(BENCH_DIR)/$(SRC_DRIVER)
	@echo Building $@ \($(KERNEL)\) . . .
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/kernel.o: $(KERNEL_DIR)/$(SRC_KERNEL)
	@echo Building $@ \($(KERNEL)\) . . .
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@echo Building $@ . . .
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

check: $(OBJS_COMMON) $(BUILD_DIR)/driver_check.o
	@echo Building $@ . . .
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)


measure: $(OBJS_MEASURE) $(BUILD_DIR)/driver.o
	@echo Building $@ . . .
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS) 

clean:
	rm -rf ./build check calibrate measure
