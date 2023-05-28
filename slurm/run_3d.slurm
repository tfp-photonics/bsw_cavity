#!/bin/bash
#SBATCH --job-name=bsw_cavity_3d
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%j_%x.log
#SBATCH --partition=accelerated

source $(conda info --base)/etc/profile.d/conda.sh
conda activate bsw_cavity

srun -n 76 --mpi=pmi2 --hint=nomultithread python3 ../evaluation/reference_3d.py
