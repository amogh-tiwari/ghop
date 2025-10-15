#!/bin/bash
#SBATCH --job-name=TravailGPU                   # name of job
#SBATCH --output=TravailGPU%j.out               # output file (%j = job ID)
#SBATCH --error=TravailGPU%j.err                # error file (%j = job ID)
#SBATCH --constraint=v100-16g                   # reserve GPUs with 16 GB of RAM
#SBATCH --nodes=1                               # reserve 1 node
#SBATCH --ntasks=1                              # reserve 4 tasks (or processes)
#SBATCH --gres=gpu:1                            # reserve 4 GPUs
#SBATCH --cpus-per-task=6                       # reserve 10 CPUs per task (and associated memory)
#SBATCH --time=01:00:00                         # maximum allocation time "(HH:MM:SS)"
#SBATCH --qos=qos_gpu-dev                       # QoS
#SBATCH --hint=nomultithread                    # deactivate hyperthreading
#SBATCH --account=tuy@v100                      # V100 accounting

module purge                                    # purge modules inherited by default
conda deactivate                                # deactivate environments inherited by default

module load miniforge/24.9.0
conda activate ghop

cd $HOME/projects/ghop

set -x                                          # activate echo of launched commands
srun python -m preprocess.make_grasp_grab --start-idx 1 --end-idx 2 # execute script


