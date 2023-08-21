
# Matrix Multiply Benchmark
Matrix Multiply Benchmark on GPU and CPU using OpenMP, OpenACC, HIP, rocBLAS, CUDA, cuBLAS, MKL and ARMpl


## Author

- [@Kass](https://www.github.com/Kassouley) 

![Kass](https://cdn.discordapp.com/attachments/705826516520665191/1116698582557397062/canvas100.png)


## Installation

No particulary installation needed.

Just run :
```bash
./setup.sh all
```

And build with :
```bash
./setup.sh
make measure KERNEL=[KERNEL_NAME] METRIC=[RDTSC-Cycles|GFLOPS] GPU=[NVIDIA|AMD]
make check KERNEL=[KERNEL_NAME] GPU=[NVIDIA|AMD]
```

KERNEL_NAME should be in uppercase.

METRIC is optional (GFLOPS by default)

GPU is optional (AMD by default)

Then run with :
```bash
./measure <m> <n> <k> <nb warmup> <nb rep>
./check <m> <n> <k> [file name]
```

- m, n k is the size of the matrices for the multiplication (c[mxp] = a[mxn] * b[nxp])
- nb warmup is the number of warmup before starting the bench
- nb rep is the number of repetitions to dampen the accuracy of the timer
- file name is an outfile
    
## Code Features

- Shows us the performance of a kernel in GFLOPS / millisecond or RDTSC-Cycles
- Benchmark on CPU using OpenMP and CBLAS
- Benchmark on CPU using ARMpl (on ARM)
- Benchmark on GPU using OpenMP with and without the data transfer (on NVIDIA & AMD)
- Benchmark on GPU using OpenACC with and without the data transfer (on NVIDIA)
- Benchmark on GPU using HIP with and without the data transfer (on AMD)
- Benchmark on GPU using rocBLAS with and without the data transfer (on AMD)
- Benchmark on GPU using CUDA with and without the data transfer (on NVIDIA)
- Benchmark on GPU using cuBLAS with and without the data transfer (on NVIDIA)
- Checker for all these kernels

## Script Features

### gpuXmm.sh

This script enables three actions:

- Check: Initiates the 'check' program multiple times across various kernels, with customizable sizes and precision settings.

- Measure: Launches several benchmarks on different kernels, with adjustable sizes, precision settings, and additional options such as generating graphs depicting the run results or altering the result metric.

- Rank Update: Conducts multiple benchmarks on diverse kernels, where the matrix size (for this matrix multiply : A [mxk] x B[kxm] = C [mxm]) evolves according to this pattern: for each value of k ranging from 1 to 32, m varies from 500 to 3000 in increments of 100. This command also includes options like generating graphs based on the obtained results.

Usage: ./gpuXmm.sh [cmd] {options} [args]"

cmd: check, measure, rank-update 

### Python script

All python scripts are used in gpuXmm.sh but these scripts can be use easily by user call.

check.py :
- take in argument two output file and a matrix size
- Check if the two output files are the same and if not, give the max error between these two files

plot_gen_measure.py :
- take in argument a metric format ("GFLOPS/s", "RDTSC-Cycle" or "Time (ms)"), an csv like output file from the measure script and a file name (.png)
- Generate a graph based on benchmark outputs from the measure script

plot_gen_rankupdate.py :
- take in argument an output file from the rank-update script and a file name (.png)
- Generate a graph based on rank-update outputs from the rank-update script


## Kernel List

The kernels depend on the availability of the corresponding APIs on your machine. For example, if you have HIP installed on your machine, you can utilize the HIP kernel. The available kernels must be listed in the 'kernels_available.config' file so that they can be recognized by the 'gpuXmm.sh' script. The 'setup.sh' script typically fill the valid kernels in the file if the corresponding API is installed on your machine. However, if it fails to detect them, you can manually write them into the file.

All available kernels :

- basis 
- cpu_omp 
- cblas 
- armpl
- mkl (note : multi-core)
- gpu_omp 
- gpu_omp_wo_dt 
- hip 
- hip_wo_dt 
- rocblas 
- rocblas_wo_dt
- acc 
- acc_wo_dt 
- cuda 
- cuda_wo_dt 
- cublas
- cublas_wo_dt 


## Usage/Examples

By using the script :

```bash
./gpuXmm measure {options} -mXX -nXX -pXX <kernel>
```

Example :
```bash
./gpuXmm measure -g BASIS hip rocblas_wo_dt -m1000 -vr
```
will run a 3 benchmark (RDTSC Cycles metric) of the kernel basis hip and rocblas_wo_dt for a 1000x1000 matrix in verbose and will generate a graph of the result

```bash
./gpuXmm measure -m100 -p26 -a
```
will run all available kernels (GFLOPS/s metric) for c[100x100] = a[100x26] * b[26x100]

Use the '-h' option for more information

```bash
./gpuXmm check {options} -mXX -nXX -pXX <kernel>
```
```bash
./gpuXmm check -m100 -a
```
will check all available kernels for a 100x100 matrix and compare it with the basis kernel

Use the '-h' option for more information

```
.......''',,,',;::ccccc:;,'............ 
........''';cldkO000KK00Oxoc;'..........
''''''..',cxO0000KKKKKK00000kdc'........
,,,,,,,;lk0Ooccd0KKKKKKK0klcokOd:'......
,,,,,,:x00d'   .:OKKKKK0o.   .lOkl'.....
''''';x00x'     .oK0000d.     .lOko'....
.''',o000l       :00000c       ;kkkl....
....:OK00:       :0K000c       ;kkkd,...
 ..'l0000l      .oKKKKKo.      :kkkx, ..
   .l000Kk,    .c0KKKKKO;     'dkkkk;  .
   .:O0000kl;;:d0KK00000kc'.':xkkkkx;...
....;k0000OOOO00000000000Okxxkkkkkkd,...
....'lOOOOOOOOOO0000000OOOOkxxkkxxxc....
.....,dOOkkOOOOOOO0000OOkkkkkxxxxxl.....
......,dkkOOOOOOOOkxdxOOkkkkkxxxdc.... .
........cxkOkkkkkx:..;dxkkkkOkxo,. .....
.........':odxdddolccloooddddo;.     ...  
............,;cllloooolllc:,..  ........        
..................'''.....  ............        
```
