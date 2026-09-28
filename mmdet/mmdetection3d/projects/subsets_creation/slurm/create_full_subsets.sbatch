#!/bin/bash
#SBATCH --partition=general # request partition
#SBATCH --qos=medium # Quality of service, short default
#SBATCH --job-name=cmt-splits
#SBATCH --output=/YOUR_PATH_HERE/logs/cmt_data_prep/default_splits.out
#SBATCH --error=/YOUR_PATH_HERE/logs/cmt_data_prep/default_splits.err
#SBATCH --time=01:30:00 # 0:01:00 is one minute, max 4hrs for short QoS, 36hrs for medium. One takes 1:04hrs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G

# define image and work dirs
export APPTAINER_IMAGE=/YOUR_WORKDIR_PATH_HERE/image/mmdet3d_v1rc5.sif

# temporary image dir to install openpyxl
mkdir -p /YOUR_PATH_HERE/mmdet/mmdetection3d/tmpfs

# REMEMBER TO EDIT HERE THE PYTHON FILE IF MOVING PLACES
echo "===GOING INTO CONTAINER==="
apptainer exec --cleanenv --overlay tmpfs \
--bind "$HOME/fedcka:/workspace/fedcka"   \
--bind "/YOUR_PATH_HERE:/YOUR_PATH_HERE"   \
--bind "/YOUR_PATH_HERE/dataset:/YOUR_PATH_HERE/dataset"   \
"$APPTAINER_IMAGE" \
bash -lc "
pip install openpyxl && 
python -c 'import openpyxl' &&
cd /workspace/mmdet/mmdetection3d && 
python projects/subsets_creation/full_subsets_nuscenes.py
"
