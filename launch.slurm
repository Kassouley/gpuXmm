#!/bin/bash
#SBATCH -N 2
#SBATCH -n 160
#SBATCH --gres=gpu:1
#SBATCH -p small
#SBATCH --ntasks-per-node=80
#SBATCH --time=03:00:00

workdir=$PWD/slurm/${SLURM_JOBID}
mkdir ${workdir}

cmd=$@

echo "Chargement des modules"
module purge
#module load arm
#module load acfl/23.04

module load nvidia
module load nvhpc/23.1
module load python-tools/3.10.9
# module load gnu/12.2.0
module load openblas/gnu/0.3.21
echo "Chargement des modules terminés"

eval $cmd

jobinfo "${SLURM_JOBID}"



