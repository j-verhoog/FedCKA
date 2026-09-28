#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --job-name=fedckarho040
#SBATCH --output=/YOUR_WORKDIR_PATH_HERE/logs/cmt/fs_cka_job%j_rank%t.out
#SBATCH --error=/YOUR_WORKDIR_PATH_HERE/logs/cmt/fs_cka_job%j_rank%t.err
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -e # stop if something fails

source /YOUR_PATH_HERE/mmdet/mmdetection3d/projects/cmt_40_epoch/experiments_config.sh

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}
export WORK=/YOUR_WORKDIR_PATH_HERE/projects/$WANDB_PROJECT/$RUN_NAME
mkdir -p "$WORK"

DATASET_A="$DAY_CLEAR_DATAROOT"
DATASET_B="$DAY_RAIN_DATAROOT"
DATASET_C="$DAY_CLEAR_DATAROOT_SING"
DATASET_D="$NIGHT_CLEAR_DATAROOT"
DATASET_E="$NIGHT_RAIN_DATAROOT"

# ==========================================
# --- AUTO-DETECT RESUME STATE ---
# ==========================================
for (( i=1; i<=NUM_ROUNDS; i++ )); do
    if [ -d "$WORK/round_$i" ]; then
        LATEST_ROUND=$i
    fi
done

if [ "$LATEST_ROUND" -gt 0 ]; then
    STOP_EPOCH=$((LATEST_ROUND * EPOCHS_PER_ROUND))
    
    if [ -f "$WORK/round_${LATEST_ROUND}/merged_E.pth" ]; then
        # The round completely finished merging
        START_ROUND=$((LATEST_ROUND + 1))
        START_MODEL="A"
    elif [ -f "$WORK/round_${LATEST_ROUND}/ModelE/epoch_${STOP_EPOCH}.pth" ]; then
        # Model E finished, but merging timed out. 
        # Setting START_MODEL="merge" and bumping the round forces runner.sh to merge the previous round first.
        START_ROUND=$((LATEST_ROUND + 1))
        START_MODEL="merge"
    elif [ -f "$WORK/round_${LATEST_ROUND}/ModelD/epoch_${STOP_EPOCH}.pth" ]; then
        START_ROUND=$LATEST_ROUND
        START_MODEL="E"
    elif [ -f "$WORK/round_${LATEST_ROUND}/ModelC/epoch_${STOP_EPOCH}.pth" ]; then
        START_ROUND=$LATEST_ROUND
        START_MODEL="D"
    elif [ -f "$WORK/round_${LATEST_ROUND}/ModelB/epoch_${STOP_EPOCH}.pth" ]; then
        START_ROUND=$LATEST_ROUND
        START_MODEL="C"
    elif [ -f "$WORK/round_${LATEST_ROUND}/ModelA/epoch_${STOP_EPOCH}.pth" ]; then
        START_ROUND=$LATEST_ROUND
        START_MODEL="B"
    else
        START_ROUND=$LATEST_ROUND
        START_MODEL="A"
    fi
fi

echo "=== Auto-detected resume state: START_ROUND=$START_ROUND | START_MODEL=$START_MODEL ==="
# ==========================================

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
    local ROUND=$4
    local GLOBAL_EPOCHS=$5
    local STOP_EPOCH=$6
    local IS_RESUME=$7
    local WANDB_RUN_ID=$8
    
    local WORK_DIR="$WORK/round_${ROUND}/${MODEL_NAME}"
    local CFG_LOAD_ARG=""
    local RESUME_ARG=""
    
    if [ "$IS_RESUME" == "true" ]; then
        # In rounds > 1, we RESUME. This loads optimizer state and epoch count.
        RESUME_ARG="--resume-from ${CHECKPOINT}"
    else
        # In round 1, we LOAD weights only (backbone). Optimizer starts fresh.
        CFG_LOAD_ARG="load_from='${CHECKPOINT}'"
    fi

    echo "=== ROUND $ROUND | Training $MODEL_NAME | Stop Epoch: $STOP_EPOCH (Global: $GLOBAL_EPOCHS) ==="
    
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
            total_epochs=$GLOBAL_EPOCHS \
            custom_hooks.1.stop_epoch=$STOP_EPOCH
      '
    "
}
# --- MERGE FUNCTION ---
run_merge () {
    local CKPT_A=$1
    local CKPT_B=$2
    local CKPT_C=$3
    local CKPT_D=$4
    local CKPT_E=$5
    local OUT_A=$6
    local OUT_B=$7
    local OUT_C=$8
    local OUT_D=$9
    local OUT_E=${10}
    local METHOD=${11}
    
    echo "=== Merging Models using $METHOD ==="
    
    # Base command string
    local MERGE_CMD="python projects/cmt_40_epoch/tools/modular_merging.py \
            --inputs $CKPT_A $CKPT_B $CKPT_C $CKPT_D $CKPT_E \
            --outputs $OUT_A $OUT_B $OUT_C $OUT_D $OUT_E \
            --weight-a $SIZE_A --weight-b $SIZE_B --weight-c $SIZE_C --weight-d $SIZE_D --weight-e $SIZE_E \
            --method $METHOD \
            --config projects/configs/cmt_voxel01_vov_1600x640_cbgs_syncbn.py \
            --select-ratio 0.02 \
            --max-sparsity 0.4"
            
    # Append CKA specific arguments if the method is fedselect_cka
    if [ "$METHOD" = "fedselect_cka" ]; then
        MERGE_CMD="$MERGE_CMD \
            --runner-path projects/cmt_40_epoch/fedcka/scripts_cmt_copy/runner.py \
            --data-dir-A $DATASET_A \
            --data-dir-B $DATASET_B \
            --data-dir-C $DATASET_C \
            --data-dir-D $DATASET_D \
            --data-dir-E $DATASET_E \
            --modality lidar_camera"
    fi

    apptainer exec --nv --cleanenv \
      --bind "$HOME/mmdet:/workspace/mmdet" \
      --bind "$WORK:/workspace/work_dirs" \
      --bind "/YOUR_WORKDIR_PATH_HERE:/YOUR_WORKDIR_PATH_HERE" \
      --bind "/YOUR_NUSCENES_PATH_HERE:/YOUR_NUSCENES_PATH_HERE" \
      "$APPTAINER_IMAGE" \
      bash -lc "
        cd /workspace/mmdet/mmdetection3d && \
        export PYTHONPATH=/workspace/mmdet/mmdetection3d:/workspace/mmdet:\$PYTHONPATH && \
        $MERGE_CMD
      "
}

# ==========================================
# MAIN LOOP
# ==========================================

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

if [ "$START_ROUND" -eq 1 ]; then
    echo "Starting fresh run from Round 1"
    MODEL_A="$PRETRAINED_BACKBONE"
    MODEL_B="$PRETRAINED_BACKBONE"
    MODEL_C="$PRETRAINED_BACKBONE"
    MODEL_D="$PRETRAINED_BACKBONE"
    MODEL_E="$PRETRAINED_BACKBONE"
else
    echo "Resuming run from Round $START_ROUND"
    PREV_ROUND=$((START_ROUND - 1))
    MODEL_A="$WORK/round_${PREV_ROUND}/merged_A.pth"
    MODEL_B="$WORK/round_${PREV_ROUND}/merged_B.pth"
    MODEL_C="$WORK/round_${PREV_ROUND}/merged_C.pth"
    MODEL_D="$WORK/round_${PREV_ROUND}/merged_D.pth"
    MODEL_E="$WORK/round_${PREV_ROUND}/merged_E.pth"
fi

# CALCULATE GLOBAL TOTAL ONCE
# This ensures the LR scheduler sees the full picture (e.g., 20 epochs) from Round 1
GLOBAL_MAX_EPOCHS=$((NUM_ROUNDS * EPOCHS_PER_ROUND))

for ((i=START_ROUND; i<=NUM_ROUNDS; i++)); do
    # This is where we want to STOP in this specific round
    CURRENT_STOP_EPOCH=$((i * EPOCHS_PER_ROUND))
    
    # --- PHASE merging DETERMINATION ---
    if [ "$i" -le 10 ]; then
        CURRENT_METHOD="fedavg"
    else
        CURRENT_METHOD="fedselect_cka"
    fi
    
    echo "-------------------------------------"
    echo "       STARTING $CURRENT_METHOD ROUND $i      "
    echo "       Scheduler Horizon: $GLOBAL_MAX_EPOCHS epochs"
    echo "       Stop Training At:  $CURRENT_STOP_EPOCH epochs"
    echo "-------------------------------------"

    if [ "$i" -eq 1 ]; then
        IS_RESUME="false"
    else
        IS_RESUME="true"
    fi

    if [ "$START_MODEL" != "A" ]; then
        echo "Starting Round $i from Model $START_MODEL , then resuming normal training after that."
        if [ "$START_MODEL" = "B" ]; then
            CKPT_A="$WORK/round_${i}/ModelA/epoch_${CURRENT_STOP_EPOCH}.pth"
            run_training "ModelB" "$DATASET_B" "$MODEL_B" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_B"
            CKPT_B="$WORK/round_${i}/ModelB/epoch_${CURRENT_STOP_EPOCH}.pth"
            run_training "ModelC" "$DATASET_C" "$MODEL_C" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_C"
            CKPT_C="$WORK/round_${i}/ModelC/epoch_${CURRENT_STOP_EPOCH}.pth"
            run_training "ModelD" "$DATASET_D" "$MODEL_D" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_D"
            CKPT_D="$WORK/round_${i}/ModelD/epoch_${CURRENT_STOP_EPOCH}.pth"
            run_training "ModelE" "$DATASET_E" "$MODEL_E" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_E"
            CKPT_E="$WORK/round_${i}/ModelE/epoch_${CURRENT_STOP_EPOCH}.pth"  
        elif [ "$START_MODEL" = "C" ]; then
            CKPT_A="$WORK/round_${i}/ModelA/epoch_${CURRENT_STOP_EPOCH}.pth"
            CKPT_B="$WORK/round_${i}/ModelB/epoch_${CURRENT_STOP_EPOCH}.pth"    
            run_training "ModelC" "$DATASET_C" "$MODEL_C" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_C"
            CKPT_C="$WORK/round_${i}/ModelC/epoch_${CURRENT_STOP_EPOCH}.pth"
            run_training "ModelD" "$DATASET_D" "$MODEL_D" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_D"
            CKPT_D="$WORK/round_${i}/ModelD/epoch_${CURRENT_STOP_EPOCH}.pth"
            run_training "ModelE" "$DATASET_E" "$MODEL_E" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_E"
            CKPT_E="$WORK/round_${i}/ModelE/epoch_${CURRENT_STOP_EPOCH}.pth"
        elif [ "$START_MODEL" = "D" ]; then
            CKPT_A="$WORK/round_${i}/ModelA/epoch_${CURRENT_STOP_EPOCH}.pth"
            CKPT_B="$WORK/round_${i}/ModelB/epoch_${CURRENT_STOP_EPOCH}.pth"    
            CKPT_C="$WORK/round_${i}/ModelC/epoch_${CURRENT_STOP_EPOCH}.pth"
            run_training "ModelD" "$DATASET_D" "$MODEL_D" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_D"
            CKPT_D="$WORK/round_${i}/ModelD/epoch_${CURRENT_STOP_EPOCH}.pth"
            run_training "ModelE" "$DATASET_E" "$MODEL_E" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_E"
            CKPT_E="$WORK/round_${i}/ModelE/epoch_${CURRENT_STOP_EPOCH}.pth"
        elif [ "$START_MODEL" = "E" ]; then
            CKPT_A="$WORK/round_${i}/ModelA/epoch_${CURRENT_STOP_EPOCH}.pth"
            CKPT_B="$WORK/round_${i}/ModelB/epoch_${CURRENT_STOP_EPOCH}.pth"    
            CKPT_C="$WORK/round_${i}/ModelC/epoch_${CURRENT_STOP_EPOCH}.pth"
            CKPT_D="$WORK/round_${i}/ModelD/epoch_${CURRENT_STOP_EPOCH}.pth"
            run_training "ModelE" "$DATASET_E" "$MODEL_E" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_E"
            CKPT_E="$WORK/round_${i}/ModelE/epoch_${CURRENT_STOP_EPOCH}.pth"
        elif [ "$START_MODEL" = "merge" ]; then
            # Calculate correct previous round and its stop epoch
            PREV_RND=$((i - 1))
            PREV_STOP_EPOCH=$((CURRENT_STOP_EPOCH - EPOCHS_PER_ROUND))
            
            # --- NEW: Determine the correct method for the previous round ---
            if [ "$PREV_RND" -le 10 ]; then
                PREV_METHOD="fedavg"
            else
                PREV_METHOD="fedselect_cka"
            fi
            # ----------------------------------------------------------------
            
            CKPT_A="$WORK/round_${PREV_RND}/ModelA/epoch_${PREV_STOP_EPOCH}.pth"
            CKPT_B="$WORK/round_${PREV_RND}/ModelB/epoch_${PREV_STOP_EPOCH}.pth"
            CKPT_C="$WORK/round_${PREV_RND}/ModelC/epoch_${PREV_STOP_EPOCH}.pth"
            CKPT_D="$WORK/round_${PREV_RND}/ModelD/epoch_${PREV_STOP_EPOCH}.pth"
            CKPT_E="$WORK/round_${PREV_RND}/ModelE/epoch_${PREV_STOP_EPOCH}.pth"
            
            # 3. Aggregate and Save 5 Distinct Models for the PREVIOUS round
            MERGED_A="$WORK/round_${PREV_RND}/merged_A.pth"
            MERGED_B="$WORK/round_${PREV_RND}/merged_B.pth"
            MERGED_C="$WORK/round_${PREV_RND}/merged_C.pth"
            MERGED_D="$WORK/round_${PREV_RND}/merged_D.pth"
            MERGED_E="$WORK/round_${PREV_RND}/merged_E.pth"
            
            # --- UPDATED: Pass $PREV_METHOD instead of "fedavg" ---
            run_merge "$CKPT_A" "$CKPT_B" "$CKPT_C" "$CKPT_D" "$CKPT_E" \
                    "$MERGED_A" "$MERGED_B" "$MERGED_C" "$MERGED_D" "$MERGED_E" "$PREV_METHOD"
            # ------------------------------------------------------
            
            # Now run Round 2 training (IS_RESUME="true" will use the newly created MERGED checkpoints)
            run_training "ModelA" "$DATASET_A" "$MODEL_A" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_A"
            CKPT_A="$WORK/round_${i}/ModelA/epoch_${CURRENT_STOP_EPOCH}.pth"

            run_training "ModelB" "$DATASET_B" "$MODEL_B" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_B"
            CKPT_B="$WORK/round_${i}/ModelB/epoch_${CURRENT_STOP_EPOCH}.pth"

            run_training "ModelC" "$DATASET_C" "$MODEL_C" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_C"
            CKPT_C="$WORK/round_${i}/ModelC/epoch_${CURRENT_STOP_EPOCH}.pth"

            run_training "ModelD" "$DATASET_D" "$MODEL_D" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_D"
            CKPT_D="$WORK/round_${i}/ModelD/epoch_${CURRENT_STOP_EPOCH}.pth"

            run_training "ModelE" "$DATASET_E" "$MODEL_E" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_E"
            CKPT_E="$WORK/round_${i}/ModelE/epoch_${CURRENT_STOP_EPOCH}.pth"
        else
            echo "Invalid START_MODEL specified: $START_MODEL. Must be one of A, B, C, D, E."
            exit 1
        fi
        START_MODEL="A" # Reset to normal training for next rounds



    else
        run_training "ModelA" "$DATASET_A" "$MODEL_A" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_A"
        CKPT_A="$WORK/round_${i}/ModelA/epoch_${CURRENT_STOP_EPOCH}.pth"

        run_training "ModelB" "$DATASET_B" "$MODEL_B" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_B"
        CKPT_B="$WORK/round_${i}/ModelB/epoch_${CURRENT_STOP_EPOCH}.pth"

        run_training "ModelC" "$DATASET_C" "$MODEL_C" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_C"
        CKPT_C="$WORK/round_${i}/ModelC/epoch_${CURRENT_STOP_EPOCH}.pth"

        run_training "ModelD" "$DATASET_D" "$MODEL_D" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_D"
        CKPT_D="$WORK/round_${i}/ModelD/epoch_${CURRENT_STOP_EPOCH}.pth"

        run_training "ModelE" "$DATASET_E" "$MODEL_E" "$i" "$GLOBAL_MAX_EPOCHS" "$CURRENT_STOP_EPOCH" "$IS_RESUME" "$WANDB_ID_E"
        CKPT_E="$WORK/round_${i}/ModelE/epoch_${CURRENT_STOP_EPOCH}.pth"
    fi

    # 3. Aggregate and Save 5 Distinct Models
    MERGED_A="$WORK/round_${i}/merged_A.pth"
    MERGED_B="$WORK/round_${i}/merged_B.pth"
    MERGED_C="$WORK/round_${i}/merged_C.pth"
    MERGED_D="$WORK/round_${i}/merged_D.pth"
    MERGED_E="$WORK/round_${i}/merged_E.pth"
    
    run_merge "$CKPT_A" "$CKPT_B" "$CKPT_C" "$CKPT_D" "$CKPT_E" \
              "$MERGED_A" "$MERGED_B" "$MERGED_C" "$MERGED_D" "$MERGED_E" "$CURRENT_METHOD"
    
    # --- FEDSELECT HANDOVER LOGIC ---
    # If we just finished Round 10, save Model A (since all are identical after FedAvg) 
    # as the baseline global model for FedSelect in Round 11
    if [ "$i" -eq 10 ]; then
        echo "Transitioning to FedSelect next round. Saving baseline global model..."
        mkdir -p "$WORK/fedselect_states"
        cp "$MERGED_A" "$WORK/fedselect_states/global_model.pth"
    fi

    # Setup for next round
    MODEL_A="$MERGED_A"
    MODEL_B="$MERGED_B"
    MODEL_C="$MERGED_C"
    MODEL_D="$MERGED_D"
    MODEL_E="$MERGED_E"
    
    echo "Round $i complete. 5 distinct merged models saved."
done

# Cleanup
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 rm -rf "$TMP_BASE"