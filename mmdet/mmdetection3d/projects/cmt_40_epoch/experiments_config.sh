#!/bin/bash

# ==========================================
# 1. GLOBAL VARIABLES (Shared across all runs)
# ==========================================
export WANDB_PROJECT="cmt_40_epoch"

# --- FEDERATED SETTINGS ---
export NUM_ROUNDS=40       # approx 2hour per round of 1 Epoch on 2 A40
export EPOCHS_PER_ROUND=1 
export SAMPLES_PER_GPU=16

# dataset sizes for each client (A, B, C, D, E)
# since each has 39, 40, 41 samples for each scene, the scene sizes are approximately the same as the dataset sizes
export SIZE_A=265
export SIZE_B=125
export SIZE_C=226
export SIZE_D=71
export SIZE_E=13

DAY_CLEAR_DATAROOT="/YOUR_WORKDIR_PATH_HERE/datasets/cmt_subsets/Default_NoFair_SingleClient/boston_day_clear"
DAY_RAIN_DATAROOT="/YOUR_WORKDIR_PATH_HERE/datasets/cmt_subsets/Default_NoFair_SingleClient/boston_day_rain"
DAY_CLEAR_DATAROOT_SING="/YOUR_WORKDIR_PATH_HERE/datasets/cmt_subsets/Default_NoFair_SingleClient/singapore_day_clear"
NIGHT_CLEAR_DATAROOT="/YOUR_WORKDIR_PATH_HERE/datasets/cmt_subsets/Default_NoFair_SingleClient/singapore_night_clear"
NIGHT_RAIN_DATAROOT="/YOUR_WORKDIR_PATH_HERE/datasets/cmt_subsets/Default_NoFair_SingleClient/singapore_night_rain"

# Initial backbone weights (only used in Round 1)
PRETRAINED_BACKBONE="ckpts/nuim_r50.pth"

# --- STANDARD SETUP ---
export WANDB_API_KEY=$(awk '$1=="machine" && $2=="api.wandb.ai" {found=1}found && $1=="password" {print $2; exit}' "$HOME/.netrc")
export APPTAINER_IMAGE=/YOUR_WORKDIR_PATH_HERE/image/mmdet3d_v1rc5.sif
export CONTINUE_WANDB_RUN=true
export START_ROUND=1
export START_MODEL='A'
export LATEST_ROUND=0

# ==========================================
# 2. EXPERIMENT-SPECIFIC VARIABLES
# ==========================================
case "$SLURM_JOB_NAME" in
    "fedMC")
        export RUN_NAME="FedMC"
        ;;
        
    "PCGrad")
        export RUN_NAME="PCGRAD"
        ;;
    
    "base-sep")
        export RUN_NAME="NoComm"
        ;;

    "fs-40p")
        export RUN_NAME="FedSelect"
        ;;

    "fedRep")
        export RUN_NAME="FedRep"
        ;;

    "feddyn")
        export RUN_NAME="FedDyn"
        ;;

    "fedBN")
        export RUN_NAME="FedBN"
        ;;
    
    "fedavg")
        export RUN_NAME="FedAvg"
        ;;
    
    "fedckatau005")
        export RUN_NAME="FedSelect_CKA_tau005"
        ;;
    "fedckatau025")
        export RUN_NAME="FedSelect_CKA_tau025"
        ;;
    "fedckatau050")
        export RUN_NAME="FedSelect_CKA_tau050"
        ;;
    "fedckatau075")
        export RUN_NAME="FedSelect_CKA_tau075"
        ;;
    "fedckarho001")
        export RUN_NAME="FedSelect_CKA_rho001"
        ;;
    "fedckarho010")
        export RUN_NAME="FedSelect_CKA_rho010"
        ;;
    "fedckarho040")
        export RUN_NAME="FedSelect_CKA_rho040"
        ;;
    
    *)
        echo "ERROR: Unknown SLURM_JOB_NAME ($SLURM_JOB_NAME). No config found."
        exit 1
        ;;
esac