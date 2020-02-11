#!/bin/bash
#SBATCH --job-name=fw
#SBATCH --error=fw.err
#SBATCH --output=fw.out
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --partition=veryshortrun
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:16
#SBATCH --exclusive

module load pgi
module load openmpi

export OMP_PROC_BIND=false
export MP_BIND=false
export ACC_NUM_CORES=4

pgcc -I${MPIINC} -L${MPILIB} -lmpi -acc -ta=multicore -Minfo=all -o accmpi main.c
# srun --cpu_bind=v,sockets --mem_bind=v,local ./accmpi
srun ./accmpi
