# MMDetection3D Setup

Once your Singularity/Apptainer `.sif` image container is ready from `/image`, complete the repository setup below to run the project experiments.

---

## 1. Environment & Codebase Setup

Clone the matching base version of MMDetection3D to populate any missing upstream files without overwriting custom project code:

```bash
# Clone compatible upstream release to a temporary directory
git clone --depth 1 --branch v1.0.0rc5 [https://github.com/open-mmlab/mmdetection3d.git](https://github.com/open-mmlab/mmdetection3d.git) /tmp/mmdet3d_temp

# Copy missing files while strictly preserving all custom files and folders
cd /YOUR_PATH_HERE/fedcka/mmdet
rsync -av --ignore-existing /tmp/mmdet3d_temp/ ./

# Clean up temporary directory
rm -rf /tmp/mmdet3d_temp

```

---

## 2. Checkpoints

Create the checkpoint directory:

```bash
mkdir -p /YOUR_PATH_HERE/fedcka/mmdet/mmdetection3d/ckpts

```

Place the required pretrained weights inside `/ckpts`:

* **`nuim_r50.pth`** (Download from Hugging Face) – Required base pretrained checkpoint.
* **`fcos3d_vovnet_imgbackbone-remapped.pth`** – Required only if reproducing the full CMT configuration.

---

## 3. Running Experiments

With the environment and weights in place, proceed to the project configs and runner scripts:

* See `fedcka/mmdetection3d/projects/README.md` for project structure, script execution, and aggregation guidelines.

