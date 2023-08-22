#!/bin/bash
setup_output()
{
    if [[ -d "./output/" ]]; then
        rm -rf ./output/
    fi
    mkdir ./output/
    mkdir ./output/check
    mkdir ./output/graphs
    mkdir ./output/measure
    mkdir ./output/profiler
    mkdir ./output/tmp
}

setup_kernels()
{
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
}


WORKDIR=`realpath $(dirname $0)`
cd $WORKDIR

case $1 in
    all) setup_kernels ; setup_output ;;
    output) setup_output ;;
    kernels) setup_kernels ;;
    *) echo "Usage : $0 [ all | output | kernels ]" ; exit 1 ;;
esac
echo "Setup finished"

