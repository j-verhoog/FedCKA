#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --job-name=base-sep
#SBATCH --output=/YOUR_WORKDIR_PATH_HERE/logs/cmt/base-sep-40_job%j_rank%t.out
#SBATCH --error=/YOUR_WORKDIR_PATH_HERE/logs/cmt/base-sep-40_job%j_rank%t.err
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -e # stop if something fails

source /YOUR_PATH_HERE/mmdet/mmdetection3d/projects/cmt_40_epoch/experiments_config.sh

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}
export WORK=/YOUR_WORKDIR_PATH_HERE/projects/$WANDB_PROJECT/$RUN_NAME
mkdir -p "$WORK"

# Paths
DATASET_A=$DAY_CLEAR_DATAROOT
DATASET_B=$DAY_RAIN_DATAROOT
DATASET_C=$DAY_CLEAR_DATAROOT_SING
DATASET_D=$NIGHT_CLEAR_DATAROOT
DATASET_E=$NIGHT_RAIN_DATAROOT


# Create per-model sub work directories under the main WORK dir
mkdir -p "$WORK"/ModelA "$WORK"/ModelB "$WORK"/ModelC "$WORK"/ModelD "$WORK"/ModelE

# Total epochs each model should train for
GLOBAL_MAX_EPOCHS=$(( NUM_ROUNDS * EPOCHS_PER_ROUND ))
TOTAL_EPOCHS=$GLOBAL_MAX_EPOCHS

TMP_BASE="/YOUR_TMPDIR_PATH_HERE/_${SLURM_JOB_ID}"

# --- NODE SETUP ---
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 mkdir -p "$TMP_BASE"/{wandb,torch,xdg,tmp,home}
# Load required modules for your cluster here
# Load cuda  module for your cluster here

# master port queries for free ports and uses that per training run, ensuring all processes can communicate properly
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# --- TRAINING FUNCTION ---
run_training () {
    local MODEL_NAME=$1
    local DATASET_PATH=$2
    local CHECKPOINT=$3
    local GLOBAL_EPOCHS=$4
    local STOP_EPOCH=$5
    local IS_RESUME=$6
    local WANDB_RUN_ID=$7
    
    local WORK_DIR="$WORK/${MODEL_NAME}"
    local CFG_LOAD_ARG=""
    local RESUME_ARG=""
    # Ensure work dir exists
    mkdir -p "$WORK_DIR"
    
    if [ "$IS_RESUME" == "true" ]; then
        # In rounds > 1, we RESUME. This loads optimizer state and epoch count.
        RESUME_ARG="--resume-from ${CHECKPOINT}"
    else
        # In round 1, we LOAD weights only (backbone). Optimizer starts fresh.
        CFG_LOAD_ARG="load_from='${CHECKPOINT}'"
    fi

    echo "=== Training $MODEL_NAME | Stop Epoch: $STOP_EPOCH (Global: $GLOBAL_EPOCHS) ==="
    
    # Grab a fresh, guaranteed-free port for THIS specific training run
    local CURRENT_MASTER_PORT=$(apptainer exec --cleanenv "$APPTAINER_IMAGE" python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
    echo "Using Master Port: $CURRENT_MASTER_PORT"

    srun --ntasks=2 --gpus-per-task=1 bash -lc "
      apptainer exec --nv --cleanenv \
      --bind \"$HOME/mmdet:/workspace/mmdet\" \
      --bind \"$TMP_BASE:/workspace/tmp\" \
      --bind \"$WORK:/workspace/work_dirs\" \
      --bind \"/YOUR_WORKDIR_PATH_HERE:/YOUR_WORKDIR_PATH_HERE\" \
      --bind \"/YOUR_NUSCENES_PATH_HERE:/YOUR_NUSCENES_PATH_HERE\" \
      --bind \"$DATASET_PATH:/workspace/mmdet/mmdetection3d/data/nuscenes\" \
      --env SLURM_PROCID=\"\$SLURM_PROCID\" \
      --env SLURM_NTASKS=\"\$SLURM_NTASKS\" \
      --env SLURM_NODELIST=\"\$SLURM_NODELIST\" \
      --env SLURM_JOB_ID=\"\$SLURM_JOB_ID\" \
      --env TORCH_HOME=\"/workspace/tmp/torch/\$SLURM_PROCID\" \
      --env XDG_CACHE_HOME=\"/workspace/tmp/xdg/\$SLURM_PROCID\" \
      --env WANDB_DIR=\"/workspace/tmp/wandb/\$SLURM_PROCID\" \
      --env TMPDIR=\"/workspace/tmp/tmp/\$SLURM_PROCID\" \
      --env WANDB_PROJECT=\"$WANDB_PROJECT\" \
      --env WANDB_NAME=\"${RUN_NAME}_${MODEL_NAME}\" \
      --env WANDB_API_KEY=\"$WANDB_API_KEY\" \
      --env OMP_NUM_THREADS=1 \
      --env MKL_NUM_THREADS=1 \
      --env NCCL_DEBUG=INFO \
      --env NCCL_ASYNC_ERROR_HANDLING=1 \
      --env NCCL_BLOCKING_WAIT=1 \
      --env TORCH_DISTRIBUTED_DEBUG=DETAIL \
      --env MASTER_ADDR=\"$MASTER_ADDR\" \
      --env MASTER_PORT=\"$CURRENT_MASTER_PORT\" \
      --env NCCL_SOCKET_IFNAME=\"lo\" \
      --env WANDB_RUN_ID=\"$WANDB_RUN_ID\" \
      --env WANDB_RESUME=\"allow\" \
      \"$APPTAINER_IMAGE\" \
      bash -lc '
        cd /workspace/mmdet/mmdetection3d && \
        export PYTHONPATH=/workspace/mmdet/mmdetection3d:/workspace/mmdet:\$PYTHONPATH && \
        python -u tools/train.py \
        projects/configs/cmt_voxel01_vov_1600x640_cbgs_syncbn.py \
          --launcher slurm \
          --work-dir $WORK_DIR \
          --seed 0 \
          $RESUME_ARG \
          --cfg-options ${CFG_LOAD_ARG} dist_params.port=$CURRENT_MASTER_PORT \
            data.samples_per_gpu=$SAMPLES_PER_GPU \
            total_epochs=$GLOBAL_EPOCHS
      '
    "
}

# Create persistent WandB Run IDs based on CONTINUE_WANDB_RUN flag
if [ "$CONTINUE_WANDB_RUN" == "true" ]; then
    export WANDB_ID_A="${RUN_NAME}_A"
    export WANDB_ID_B="${RUN_NAME}_B"
    export WANDB_ID_C="${RUN_NAME}_C"
    export WANDB_ID_D="${RUN_NAME}_D"
    export WANDB_ID_E="${RUN_NAME}_E"
else
    export WANDB_ID_A="job_${SLURM_JOB_ID}_A"
    export WANDB_ID_B="job_${SLURM_JOB_ID}_B"
    export WANDB_ID_C="job_${SLURM_JOB_ID}_C"
    export WANDB_ID_D="job_${SLURM_JOB_ID}_D"
    export WANDB_ID_E="job_${SLURM_JOB_ID}_E"
fi


# Autodetect per-model "latest.pth" inside the work directory and set per-model resume flags.
# If a latest checkpoint exists for a model, resume from it; otherwise use the configured MODEL_*.
find_latest_checkpoint() {
    local MODEL_NAME="$1"
    local LATEST="$WORK/${MODEL_NAME}/latest.pth"
    if [ -f "$LATEST" ]; then
        echo "$LATEST"
    else
        echo ""
    fi
}

# Default to using the pretrained backbone unless a per-model latest checkpoint exists
MODEL_A="$PRETRAINED_BACKBONE"
MODEL_B="$PRETRAINED_BACKBONE"
MODEL_C="$PRETRAINED_BACKBONE"
MODEL_D="$PRETRAINED_BACKBONE"
MODEL_E="$PRETRAINED_BACKBONE"

LATEST_A=$(find_latest_checkpoint "ModelA")
LATEST_B=$(find_latest_checkpoint "ModelB")
LATEST_C=$(find_latest_checkpoint "ModelC")
LATEST_D=$(find_latest_checkpoint "ModelD")
LATEST_E=$(find_latest_checkpoint "ModelE")

if [ -n "$LATEST_A" ]; then MODEL_A="$LATEST_A"; IS_RESUME_A=true; else IS_RESUME_A=false; fi
if [ -n "$LATEST_B" ]; then MODEL_B="$LATEST_B"; IS_RESUME_B=true; else IS_RESUME_B=false; fi
if [ -n "$LATEST_C" ]; then MODEL_C="$LATEST_C"; IS_RESUME_C=true; else IS_RESUME_C=false; fi
if [ -n "$LATEST_D" ]; then MODEL_D="$LATEST_D"; IS_RESUME_D=true; else IS_RESUME_D=false; fi
if [ -n "$LATEST_E" ]; then MODEL_E="$LATEST_E"; IS_RESUME_E=true; else IS_RESUME_E=false; fi

# Helper function to skip fully trained models
check_and_run() {
    local MODEL_NAME=$1
    local DATASET_PATH=$2
    local MODEL_CKPT=$3
    local IS_RESUME_FLAG=$4
    local WANDB_ID=$5

    # Check if the final epoch checkpoint exists
    if [ -f "$WORK/${MODEL_NAME}/epoch_${TOTAL_EPOCHS}.pth" ]; then
        echo "=== Skipping $MODEL_NAME: epoch_${TOTAL_EPOCHS}.pth found. Already fully trained. ==="
    else
        run_training "$MODEL_NAME" "$DATASET_PATH" "$MODEL_CKPT" "$GLOBAL_MAX_EPOCHS" "$GLOBAL_MAX_EPOCHS" "$IS_RESUME_FLAG" "$WANDB_ID"
    fi
}

# Run training (or skip if completely finished)
check_and_run "ModelA" "$DATASET_A" "$MODEL_A" "$IS_RESUME_A" "$WANDB_ID_A"
check_and_run "ModelB" "$DATASET_B" "$MODEL_B" "$IS_RESUME_B" "$WANDB_ID_B"
check_and_run "ModelC" "$DATASET_C" "$MODEL_C" "$IS_RESUME_C" "$WANDB_ID_C"
check_and_run "ModelD" "$DATASET_D" "$MODEL_D" "$IS_RESUME_D" "$WANDB_ID_D"
check_and_run "ModelE" "$DATASET_E" "$MODEL_E" "$IS_RESUME_E" "$WANDB_ID_E"

echo "All 5 baseline models trained independently for $TOTAL_EPOCHS epochs."

# Measure GPU usage of your job (result) for all allocated GPUs
/usr/bin/nvidia-smi --query-accounted-apps='gpu_instance_id,gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' 2>/dev/null || true

# Cleanup
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 rm -rf "$TMP_BASE"