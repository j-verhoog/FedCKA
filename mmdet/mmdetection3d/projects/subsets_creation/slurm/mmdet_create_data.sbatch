#!/bin/bash
#SBATCH --partition=general # request partition
#SBATCH --qos=medium # Quality of service, short default
#SBATCH --job-name=cmt_create_data_default
#SBATCH --output=/YOUR_PATH_HERE/logs/cmt_data_prep/default_create_data.out
#SBATCH --error=/YOUR_PATH_HERE/logs/cmt_data_prep/default_create_data.err
#SBATCH --time=6:00:00 # 0:01:00 is one minute, max 4hrs for short QoS, 36hrs for medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

# --- CONFIGURATION ---
BASE_DATASETS_FOLDER="/YOUR_PATH_HERE/datasets/cmt_subsets"
APPTAINER_IMAGE="/YOUR_PATH_HERE/image/mmdet3d_v1rc5.sif"

# --- MANUAL OVERRIDE ---
# If this list is NOT empty, the script will skip discovery and ONLY process these paths.
MANUAL_OVERRIDE=(
    "$BASE_DATASETS_FOLDER/Default_NoFair_SingleClient/boston_day_clear"
    "$BASE_DATASETS_FOLDER/Default_NoFair_SingleClient/boston_day_rain"
    "$BASE_DATASETS_FOLDER/Default_NoFair_SingleClient/singapore_day_clear"
    "$BASE_DATASETS_FOLDER/Default_NoFair_SingleClient/singapore_night_clear"
    "$BASE_DATASETS_FOLDER/Default_NoFair_SingleClient/singapore_night_rain"
)

# Paths to explicitly ignore (substring matching)
MANUAL_EXCLUDE=()
    # "Exp2_TotalFair_SingleClient/boston_day_clear"
    # "Exp2_TotalFair_SingleClient/boston_day_rain"
    # "Exp2_TotalFair_SingleClient/singapore_day_clear"
    # "Exp2_TotalFair_SingleClient/singapore_night_clear"
    # "Exp2_TotalFair_SingleClient/singapore_night_rain"

# --- PHASE 1: DISCOVERY ---
TARGET_PATHS=()

if [ ${#MANUAL_OVERRIDE[@]} -gt 0 ]; then
    echo "!!! MANUAL OVERRIDE DETECTED !!!"
    TARGET_PATHS=("${MANUAL_OVERRIDE[@]}")
else
    echo "Searching for nuScenes root directories in: $BASE_DATASETS_FOLDER"
    
    # Use globbing to find v1.0-trainval at depths 2 and 3
    # Pattern 1: BASE/*/Subset/v1.0-trainval
    # Pattern 2: BASE/*/Subset/*/v1.0-trainval
    for META_DIR in "$BASE_DATASETS_FOLDER"/*/*/v1.0-trainval "$BASE_DATASETS_FOLDER"/*/*/*/v1.0-trainval; do
        
        # Check if the path actually exists (prevents literal string errors if no match)
        [ -e "$META_DIR" ] || continue
        
        NU_ROOT=$(dirname "$META_DIR")
        
        # --- EXCLUSION CHECK ---
        SKIP_DIR=false
        for EXCLUDE_PATH in "${MANUAL_EXCLUDE[@]}"; do
            if [[ "$NU_ROOT" == *"$EXCLUDE_PATH"* ]]; then
                echo "Skipping excluded directory: $NU_ROOT"
                SKIP_DIR=true
                break
            fi
        done
        
        [ "$SKIP_DIR" = true ] && continue
        # -----------------------

        # Validation: Ensure it's a real nuScenes root
        if [ -f "$META_DIR/sample.json" ] && [ -d "$NU_ROOT/samples" ]; then
            TARGET_PATHS+=("$NU_ROOT")
            echo "Found valid nuScenes root: $NU_ROOT"
        else
            echo "Skipping invalid candidate: $NU_ROOT (Missing sample.json or samples/)"
        fi
    done
fi

echo "--------------------------------------------------------"
echo "TOTAL SUBSETS TO PROCESS: ${#TARGET_PATHS[@]}"
echo "--------------------------------------------------------"

# --- PHASE 2: PROCESSING ---
TMP_BASE="/tmp/YOUR_NAME_HERE_${SLURM_JOB_ID}"
mkdir -p "$TMP_BASE"/{wandb,torch,xdg,tmp,home}

for NU_ROOT_SUBSET in "${TARGET_PATHS[@]}"; do
    SUBSET_NAME=$(basename "$NU_ROOT_SUBSET")
    
    echo ">>> PROCESSING: $SUBSET_NAME"
    echo ">>> PATH:       $NU_ROOT_SUBSET"

    apptainer exec --cleanenv \
    --bind "$HOME/mmdet:/workspace/mmdet"   \
    --bind "/YOUR_PATH_HERE:/YOUR_PATH_HERE"   \
    --bind "YOUR_PATH_HERE/datasets:YOUR_PATH_HERE/datasets"   \
    --bind "$TMP_BASE:/workspace/tmp" \
    --env TMPDIR=/workspace/tmp/tmp \
    --env TORCH_HOME=/workspace/tmp/torch   \
    --env XDG_CACHE_HOME=/workspace/tmp/xdg   \
    "$APPTAINER_IMAGE" \
    bash -lc "
    cd /workspace/mmdet/mmdetection3d && \
    python tools/create_data.py nuscenes \
      --root-path '$NU_ROOT_SUBSET' \
      --version v1.0 \
      --out-dir '$NU_ROOT_SUBSET' \
      --extra-tag nuscenes \
      --workers 16
    "

    if [ $? -eq 0 ]; then
        echo "Done with $SUBSET_NAME"
    else
        echo "!!! FAILED: $SUBSET_NAME"
    fi
    echo "--------------------------------------------------------"
done

echo "Execution complete."