#!/bin/bash

WORKDIR=`realpath $(dirname $0)`
cd $WORKDIR

if [[ -d "./output/" ]]; then
    rm -rf ./output/
fi
mkdir ./output/
mkdir ./output/check
mkdir ./output/graphs
mkdir ./output/graphs/warmup
mkdir ./output/measure
mkdir ./output/profiler
mkdir ./output/tmp

echo basis > kernels_avail.config
  
if ldconfig -p | grep -q libcblas; then
echo cblas >> kernels_avail.config
fi

if ldconfig -p | grep -q libgomp; then
    echo -e "cpu_omp\ngpu_omp\ngpu_omp_wo_dt" >> kernels_avail.config
fi

if ldconfig -p | grep -q libaccinj64; then
    echo -e "acc\nacc_wo_dt" >> kernels_avail.config
fi

if ldconfig -p | grep -q libamdhip64; then
    echo -e "hip\nhip_wo_dt" >> kernels_avail.config   
    if ldconfig -p | grep -q librocblas; then
        echo -e "rocblas\nrocblas_wo_dt" >> kernels_avail.config
    fi
fi

if command -v nvcc &>/dev/null; then
    echo -e "cuda\ncuda_wo_dt" >> kernels_avail.config
    if ldconfig -p | grep -q libcublas; then
        echo -e "cublas\ncublas_wo_dt" >> kernels_avail.config
    fi
fi


echo "Setup finished"

