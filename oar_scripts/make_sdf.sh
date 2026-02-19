#!/bin/bash
#OAR -n atiwari
#OAR -l walltime=12:0:0
#OAR -t besteffort
#OAR -t idempotent
#OAR --array 10

set -e

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
echo "$OAR_ARRAY_INDEX"

chunk_size=100
start=$(( (OAR_ARRAY_INDEX - 1) * chunk_size )) # OAR arrays are 1-indexed
end=$(( start + chunk_size ))
# -------------------- TO CHANGE --------------------- #

# -------------------- TO CHANGE --------------------- #
export PYOPENGL_PLATFORM=egl
echo Running Code ...
# python -m preprocess.make_grasp_grab --start-idx 1 --end-idx 2
python -m preprocess.make_grasp --data_name ho3d_benchmark --start_idx $start --end_idx $end
# -------------------- TO CHANGE --------------------- #
