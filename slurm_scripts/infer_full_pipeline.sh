#!/bin/bash
#SBATCH --job-name=TravailGPU                   # name of job
#SBATCH --output=TravailGPU%j.out               # output file (%j = job ID)
#SBATCH --error=TravailGPU%j.err                # error file (%j = job ID)
#SBATCH --constraint=v100-16g                   # reserve GPUs with 16 GB of RAM
#SBATCH --nodes=1                               # reserve 1 node
#SBATCH --ntasks=1                              # reserve 4 tasks (or processes)
#SBATCH --gres=gpu:1                            # reserve 4 GPUs
#SBATCH --cpus-per-task=6                       # reserve 10 CPUs per task (and associated memory)
#SBATCH --time=96:00:00                         # maximum allocation time "(HH:MM:SS)"
#SBATCH --qos=qos_gpu-t4                       # QoS
#SBATCH --hint=nomultithread                    # deactivate hyperthreading
#SBATCH --account=tuy@v100                      # V100 accounting
#SBATCH --array=0-99

module purge                                    # purge modules inherited by default
conda deactivate                                # deactivate environments inherited by default

module load miniforge/24.9.0
conda activate ghop

cd $HOME/projects/ghop

set -x                                          # activate echo of launched commands

echo "$SLURM_ARRAY_TASK_ID"

out_dir=grasps_GRAB_full_data_run_XXXX
chunk_size=2
start=$(( SLURM_ARRAY_TASK_ID * chunk_size ))
end=$(( start + chunk_size ))

echo "Job $SLURM_ARRAY_TASK_ID: range [$start, $end]"

export PYOPENGL_PLATFORM=egl
srun python -m grasp_syn_on_grab start_idx=$start end_idx=$end S=10 save_index=$out_dir vis_every_n=-1

