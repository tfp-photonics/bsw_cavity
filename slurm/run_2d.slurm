#!/bin/bash
#SBATCH --job-name=bsw_cavity_experiments
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --output=%j_%x.log
#SBATCH --partition=cpuonly

source $(conda info --base)/etc/profile.d/conda.sh
conda activate bsw_cavity

# srun --exclusive -n 1 -c 24 --mem=81000mb --mpi=pmi2 ../optimization/optimize_experiment.py 0 &
# srun --exclusive -n 1 -c 24 --mem=81000mb --mpi=pmi2 ../optimization/optimize_experiment.py 1 &
# srun --exclusive -n 1 -c 24 --mem=81000mb --mpi=pmi2 ../optimization/optimize_experiment.py 2 &
# srun --exclusive -n 1 -c 24 --mem=81000mb --mpi=pmi2 ../optimization/optimize_experiment.py 3 &
# srun --exclusive -n 1 -c 24 --mem=81000mb --mpi=pmi2 ../optimization/optimize_experiment.py 4 &
# srun --exclusive -n 1 -c 24 --mem=81000mb --mpi=pmi2 ../optimization/optimize_experiment.py 5 &
srun --exclusive -n 1 -c 24 --mem=81000mb --mpi=pmi2 ../optimization/optimize_experiment.py 6 &
srun --exclusive -n 1 -c 24 --mem=81000mb --mpi=pmi2 ../optimization/optimize_experiment.py 7 &
srun --exclusive -n 1 -c 24 --mem=81000mb --mpi=pmi2 ../optimization/optimize_experiment.py 8 &

wait
