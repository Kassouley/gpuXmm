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
echo "Setup finished"

