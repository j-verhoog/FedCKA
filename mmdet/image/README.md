# Container Image Setup

Minimal guide to building and verifying the Apptainer/Singularity container (`.sif`) required for the experiments.

## 1. Download Dependencies

Download the following required files from the provided Hugging Face and place them directly in the `/image` directory:

* Base image (Ubuntu 20.04, CUDA 11.3, GCC 10)
* Flash Attention wheel (CUDA-compatible)

## 2. Build the Image

Navigate to the image directory and build the `.sif` container using the provided definition file:

```bash
cd /YOUR_PATH_HERE/fedcka/mmdet/image
apptainer build mmdet3d_v1rc5.sif mmdet3d_v1rc5.def

```

## 3. Verify Installation

Run the verification script inside the container to ensure the environment and dependencies are correctly configured:

```bash
apptainer exec --nv --cleanenv \
  --bind /YOUR_PATH_HERE/fedcka/mmdet:/workspace/mmdet \
  mmdet3d_v1rc5.sif /usr/local/bin/verify_mmdet3d.sh

```

