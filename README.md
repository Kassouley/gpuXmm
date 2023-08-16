
# Matrix Multiply Benchmark
Matrix Multiply Benchmark on GPU and CPU using OpenMP, OpenACC, HIP, rocBLAS, CUDA, cuBLAS


## Author

- [@Kass](https://www.github.com/Kassouley) 

![Kass](https://cdn.discordapp.com/attachments/705826516520665191/1116698582557397062/canvas100.png)


## Installation

No particulary installation needed.

Just :
```bash
./setup.sh
```

And build with :
```bash
./setup.sh
make measure KERNEL=[KERNEL_NAME] CLOCK=[RDTSC|MS] GPU=[NVIDIA|AMD] CC=whateveruwant
make check KERNEL=[KERNEL_NAME] GPU=[NVIDIA|AMD] CC=whateveruwant
```

KERNEL_NAME should be in uppercase.

CLOCK is optional (MS by default)

GPU is optional (AMD by default)

CC is optional (AMD Clang on AMD and NVC on NVIDIA (NVCC for CUDA kernels) by default)

Then run with :
```bash
./measure <problem size> <nb warmup> <nb rep>
./check <problem size> [file name]
```

- problem size is the size n of an n x n matrix
- nb warmup is the number of warmup before starting the bench
- nb rep is the number of repetitions to dampen the accuracy of the timer
- file name is an outfile
- nb step is the number of step to calibrate the warmup
    
## Code Features

- Shows us the time of a kernel in millisecond or RDTSC-Cycles
- Benchmark on CPU using OpenMP and CBLAS (on NVIDIA & AMD)
- Benchmark on GPU using OpenMP with and without the data transfer (on NVIDIA & AMD)
- Benchmark on GPU using OpenACC with and without the data transfer (on NVIDIA)
- Benchmark on GPU using HIP with and without the data transfer (on AMD)
- Benchmark on GPU using rocBLAS with and without the data transfer (on AMD)
- Benchmark on GPU using CUDA with and without the data transfer (on NVIDIA)
- Benchmark on GPU using cuBLAS with and without the data transfer (on NVIDIA)
- Checker for all these kernels
- Warmup calibration for all these kernels

## Script Features

### Bash script

cmd:
    check       : check matrix multiply output
    measure     : run normal benchmarks
    rank-update : run a rank update benchmark

### Python script

check.py :
- take in argument two output file and a matrix size
- Check if the two output files are the same and if not, give the max error between these two files


graph-gen-measure.py :
- take in argument a matrix size, an output file from the measure script, a output png file and a time metric (ms or rdtsc)
- Generate a graph based on benchmark outputs from the measure script

graph-gen-warmup.py :
- take in argument a matrix size, an output file from the calibrate script, a output png file and a time metric (ms or rdtsc)
- Generate a graph based on calibration outputs from the calibrate script


## Kernel List

On AMD :

- basis 
- cpu_omp 
- cblas 
- gpu_omp 
- gpu_omp_wo_dt 
- hip 
- hip_wo_dt 
- rocblas 
- rocblas_wo_dt

On NVIDIA :

- basis 
- cpu_omp 
- cblas 
- gpu_omp 
- gpu_omp_wo_dt 
- openacc 
- openacc_wo_dt 
- cuda 
- cuda_wo_dt 
- cublas
- cublas_wo_dt 


## Documentation

[Intership report on GPU](https://www.overleaf.com/read/cjpngdgvjckd)

## Usage/Examples

By using the script :

```bash
./script/measure.sh {options} [problem size] <kernels>
```

Example :
```bash
./script/measure.sh -p BASIS hip rocblas_wo_dt 1000 -v
```
will run a 3 benchmark (RDTSC Cycles metric) of the kernel basis hip and rocblas_wo_dt for a 1000x1000 matrix in verbose and will generate a graph of the result

```bash
./script/measure.sh 100 -ma
```
will run all kernels (millisecond metric) for a 100x100 matrix

Use the '-h' option for more information

```bash
./script/check.sh {options} [problem size] <kernels>
```
```bash
./script/check.sh 100 -a
```
will check all kernels for a 100x100 matrix and compare it with the basis kernel

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