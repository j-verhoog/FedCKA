#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --job-name=Central
#SBATCH --output=/YOUR_WORKDIR_PATH_HERE/logs/cmt/centr_base_40e_%j.out
#SBATCH --error=/YOUR_WORKDIR_PATH_HERE/logs/cmt/centr_base_40e_%j.err
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -e # Good practice to fail fast

# cmt baseline downtuned Ali test
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}

# --- CONFIGURATION MATCHING FEDAVG ---
PRETRAINED_BACKBONE="ckpts/nuim_r50.pth"
export TARGET_EPOCHS=40  # Matches 40 Epochs
export SAMPLES_PER_GPU=16

export WANDB_PROJECT="cmt_40_epoch"
export WANDB_NAME="centralized_model"
export WANDB_API_KEY=$(awk '$1=="machine" && $2=="api.wandb.ai" {found=1}found && $1=="password" {print $2; exit}' "$HOME/.netrc")

export CONTINUE_WANDB_RUN=true
# Create persistent WandB Run IDs based on CONTINUE_WANDB_RUN flag
if [ "$CONTINUE_WANDB_RUN" == "true" ]; then
    export WANDB_RUN_ID="${WANDB_NAME}"
else
    export WANDB_RUN_ID="job_${SLURM_JOB_ID}"
fi

/usr/bin/scontrol show job -d "$SLURM_JOB_ID" # check sbatch working

# define image and work dirs
export APPTAINER_IMAGE=/YOUR_WORKDIR_PATH_HERE/image/mmdet3d_v1rc5.sif
export WORK=/YOUR_WORKDIR_PATH_HERE/projects/$WANDB_PROJECT/$WANDB_NAME
mkdir -p "$WORK"

# --- CHECKPOINT RESUME LOGIC ---
# If latest.pth exists, resume from it. Otherwise, load the pretrained backbone.
if [ -f "$WORK/latest.pth" ]; then
    echo "Found existing checkpoint at latest.pth! Resuming training..."
    export TRAIN_ARGS="--resume-from /workspace/work_dirs/latest.pth"
else
    echo "No checkpoint found. Starting fresh from pretrained backbone..."
    export TRAIN_ARGS="--cfg-options load_from=${PRETRAINED_BACKBONE}"
fi

# --- LOCAL DATA SETUP ---
NETWORK_DATA_ROOT="/YOUR_WORKDIR_PATH_HERE/datasets/nuscenes_cmt_full" # this needs to be the entire original nuscenes dataset
TMP_BASE="/YOUR_TMPDIR_PATH_HERE/_${SLURM_JOB_ID}"

echo "Creating tmp directories on all nodes..."
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 \
    mkdir -p "$TMP_BASE"/{wandb,torch,xdg,tmp,home}

# Measure GPU usage of your job (initialization)
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' 2>/dev/null || true)

echo "=== LOADING CUDA ==="
# Load required modules for your cluster here
# Load cuda  module for your cluster here

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))

echo "Master node is: $MASTER_ADDR"
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

echo "=== ENTERING CONTAINER =="
srun --ntasks=2 --gpus-per-task=1 bash -lc "
  apptainer exec --nv --cleanenv \
  --bind \"$HOME/mmdet:/workspace/mmdet\" \
  --bind \"$TMP_BASE:/workspace/tmp\" \
  --bind \"$WORK:/workspace/work_dirs\" \
  --bind \"/YOUR_WORKDIR_PATH_HERE:/YOUR_WORKDIR_PATH_HERE\" \
  --bind \"/YOUR_NUSCENES_PATH_HERE:/YOUR_NUSCENES_PATH_HERE\" \
  --bind \"$NETWORK_DATA_ROOT:/workspace/mmdet/mmdetection3d/data/nuscenes\" \
  --env SLURM_PROCID=\"\$SLURM_PROCID\" \
  --env SLURM_NTASKS=\"\$SLURM_NTASKS\" \
  --env SLURM_NODELIST=\"\$SLURM_NODELIST\" \
  --env SLURM_JOB_ID=\"\$SLURM_JOB_ID\" \
  --env TORCH_HOME=\"/workspace/tmp/torch/\$SLURM_PROCID\" \
  --env XDG_CACHE_HOME=\"/workspace/tmp/xdg/\$SLURM_PROCID\" \
  --env WANDB_DIR=\"/workspace/tmp/wandb/\$SLURM_PROCID\" \
  --env TMPDIR=\"/workspace/tmp/tmp/\$SLURM_PROCID\" \
  --env WANDB_PROJECT=\"$WANDB_PROJECT\" \
  --env WANDB_NAME=\"${WANDB_NAME}_\$SLURM_PROCID\" \
  --env WANDB_API_KEY=\"$WANDB_API_KEY\" \
  --env OMP_NUM_THREADS=1 \
  --env MKL_NUM_THREADS=1 \
  --env NCCL_DEBUG=INFO \
  --env NCCL_ASYNC_ERROR_HANDLING=1 \
  --env NCCL_BLOCKING_WAIT=1 \
  --env TORCH_DISTRIBUTED_DEBUG=DETAIL \
  --env MASTER_ADDR=\"$MASTER_ADDR\" \
  --env MASTER_PORT=\"$MASTER_PORT\" \
  --env NCCL_SOCKET_IFNAME=\"lo\" \
  --env TRAIN_ARGS=\"$TRAIN_ARGS\" \
  --env WANDB_RUN_ID=\"$WANDB_RUN_ID\" \
  --env WANDB_RESUME=\"allow\" \
  --env TARGET_EPOCHS=\"$TARGET_EPOCHS\" \
  --env SAMPLES_PER_GPU=\"$SAMPLES_PER_GPU\" \
  \"$APPTAINER_IMAGE\" \
  bash -lc '
    cd /workspace/mmdet/mmdetection3d && \
    export PYTHONPATH=/workspace/mmdet/mmdetection3d:/workspace/mmdet:\$PYTHONPATH && \
    python -u tools/train.py \
      projects/configs/cmt_voxel01_vov_1600x640_cbgs_syncbn.py \
      --launcher slurm \
      --work-dir /workspace/work_dirs \
      --seed 0 \
      \$TRAIN_ARGS \
      --cfg-options \
        evaluation.interval=1 \
        total_epochs=\$TARGET_EPOCHS \
        data.samples_per_gpu=\$SAMPLES_PER_GPU  \
        dist_params.port=\"$MASTER_PORT\"
  '
"

# Clean up local disk on ALL nodes
echo "Cleaning up tmp directories on all nodes..."
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 \
    rm -rf "$TMP_BASE"
    
# Measure GPU usage of your job (result)
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' 2>/dev/null || true