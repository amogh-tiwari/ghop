#!/bin/bash
#OAR -n atiwari
#OAR -l walltime=96:0:0
#OAR -p host='gpuhost20'

# -------------------- TO CHANGE --------------------- #
ROOT_DIR="/home/atiwari/projects/ghop"
CONDA_ENV_NAME="ghop"
# -------------------- TO CHANGE --------------------- #


# -------------------- STATIC  --------------------- #
CONDA_PATH="/scratch/clear/atiwari/miniconda3"
echo "$CONDA_PATH/etc/profile.d/conda.sh"
source "$CONDA_PATH/etc/profile.d/conda.sh"
echo "Initialized miniconda"
echo

echo "Python path initially: $(which python3)"
conda activate $CONDA_ENV_NAME
echo "Activated conda environment - $CONDA_ENV_NAME"
echo "Python path now: $(which python3)"
echo

echo "Working directory initially: $PWD"
cd $ROOT_DIR
echo "Moved into $ROOT_DIR"
echo "Working directory now: $PWD"
echo
# -------------------- STATIC  --------------------- #

# -------------------- TO CHANGE --------------------- #
echo Running Code ...
# python  -m grasp_syn -m grasp_dir=\${environment.data_dir}/GRAB_grasp/meshes/all S=20 save_index=grasp_GRAB
python -m grasp_syn -m grasp_dir=/scratch/clear/atiwari/datasets/grabnet_processing/sdfs/all save_index=grasp_GRAB_one_sample_per_sequence

# for item in binoculars camera fryingpan mug toothpaste wineglass; do
  # python -m grasp_syn -m grasp_dir=\${environment.data_dir}/GRAB_grasp/meshes/all S=20 save_index=grasp_GRAB/$item index=$item
# done

# -------------------- TO CHANGE --------------------- #
