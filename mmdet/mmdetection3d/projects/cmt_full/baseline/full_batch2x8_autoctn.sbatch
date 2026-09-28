#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --job-name=cmt
#SBATCH --output=/tmp/logs/cmt/paper_config_FULLjob%j_rank%t.out
#SBATCH --error=/tmp/logs/cmt/paper_config_FULLjob%j_rank%t.err
#SBATCH --time=36:00:00
#SBATCH --nodes=4              # Amount of machines
#SBATCH --ntasks=8             # Total tasks (2 per node)
#SBATCH --ntasks-per-node=2    # Spread them evenly
#SBATCH --gres=gpu:a40:2       # Request 2 GPUs per node (Total 2)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G             



set -e # Good practice to fail fast
echo "=== Starting Job ID: $SLURM_JOB_ID ==="
mkdir -p /tmp/logs/cmt

PRETRAINED_BACKBONE="ckpts/fcos3d_vovnet_imgbackbone-remapped.pth"
RESUME_FROM="/YOUR_PATH_HERE/latest.pth"
TARGET_EPOCHS=20
SAMPLES_PER_GPU=2       # 4 leads OOM on A40, 2==31GB and takes 11 days on two gpu's

# FOR NAMING STICK TO: A–Z  a–z  0–9  .  _  -
export WANDB_PROJECT="cmt"
export WANDB_NAME="paper_config_full_2x8"
export WANDB_API_KEY=$(awk '$1=="machine" && $2=="api.wandb.ai" {found=1}found && $1=="password" {print $2; exit}' "$HOME/.netrc")

/usr/bin/scontrol show job -d "$SLURM_JOB_ID" # check sbatch working

# define image and work dirs
export APPTAINER_IMAGE=/YOUR_PATH_HERE/image/mmdet3d_v1rc5.sif
export WORK=/YOUR_PATH_HERE/projects/$WANDB_PROJECT/$WANDB_NAME
mkdir -p "$WORK"

# --- LOCAL DATA SETUP ---
NETWORK_DATA_ROOT="/YOUR_PATH_HERE/datasets/nuscenes_cmt_full"
TMP_BASE="/tmp/YOUR_NAME_HERE_${SLURM_JOB_ID}"

# FIX: Use srun to create directories on ALL nodes, not just the master
echo "Creating tmp directories on all nodes..."
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 \
    mkdir -p "$TMP_BASE"/{wandb,torch,xdg,tmp,home}

# Measure GPU usage of your job (initialization)
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' 2>/dev/null || true)

echo "=== LOADING CUDA ==="
# Load required modules for your cluster here
# Load cuda  module for your cluster here

# Get the hostname of the first node in the list
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(apptainer exec --cleanenv "$APPTAINER_IMAGE" python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "Master node is: $MASTER_ADDR"
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

export WANDB_RUN_ID="c4cdf4er"

echo "=== ENTERING CONTAINER =="
srun --ntasks=8 --gpus-per-task=1 bash -lc "
  apptainer exec --nv --cleanenv \
  --bind \"$HOME/mmdet:/workspace/mmdet\" \
  --bind \"$TMP_BASE:/workspace/tmp\" \
  --bind \"$WORK:/workspace/work_dirs\" \
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
  --env WANDB_NAME=\"${WANDB_NAME}\" \
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
  --env WANDB_RUN_ID=\"$WANDB_RUN_ID\" \
  --env WANDB_RESUME=\"allow\" \
  \"$APPTAINER_IMAGE\" \
  bash -lc '
    cd /workspace/mmdet/mmdetection3d && \
    export PYTHONPATH=/workspace/mmdet/mmdetection3d:/workspace/mmdet:\$PYTHONPATH && \
    python -u tools/train.py \
      projects/configs/cmt_voxel0075_vov_1600x640_cbgs_syncbn.py \
      --launcher slurm \
      --work-dir /workspace/work_dirs \
      --cfg-options \
        evaluation.interval=1 \
        resume_from='"$RESUME_FROM"' \
        total_epochs=$TARGET_EPOCHS \
        data.samples_per_gpu=$SAMPLES_PER_GPU \
        dist_params.port='"$MASTER_PORT"'
  '
"



# Clean up local disk on ALL nodes
echo "Cleaning up tmp directories on all nodes..."
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 \
    rm -rf "$TMP_BASE"
    
# Measure GPU usage of your job (result)
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' 2>/dev/null || true





