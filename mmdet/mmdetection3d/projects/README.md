# Projects Overview

Guide to directory structure, configuration setups, and experimental replication across `mmdetection3d/projects`.

---

## Directory Structure

* **`/subsets_creation`**  
  * Generate all dataset splits for nuScenes.  
  * Produce required `.pkl` metadata files for training.

* **`/cmt`**  
  * Contains the custom training hooks for CMT.  
  * Includes client-side behavior hooks such as FedCKA, FedDyn, FedSelect, and other custom aggregation utilities.

* **`/cmt_40_epoch`**  
  * `experiments_config.sh`: Defines core environment variables required by all `.sbatch` scripts.  
  * `tools/modular_merging.py`: Implementation of all model merging strategies.  
  * Subdirectories: Each experiment contains its own dedicated directory with at least one runnable script.

* **`/cmt_full`**  
  * Complete reproduction of the original CMT baseline configuration and benchmarks.

* **`/configs`**  
  * Contains all configuration files used for the experiments.

* **`/analysis`**  
  * `/class_balance`: Reproduce the nuScenes class distribution overview.  
  * `/qualitative_results`: Generate qualitative prediction visualizations reported in the paper.

* **`/mmdet3d_plugin`**  
  * Custom extensions built on the Cross-Modal Transformer codebase.  
  * Adds FP16 support adapted and metrics adapted for small batch sizes under low-sample domain regimes.

---

## Environment & Path Configuration

Before launching jobs, replace placeholder strings across all configuration files:

* `YOUR_WORKDIR_PATH_HERE`
* `YOUR_HOME_DIR`
* `YOUR_NAME_HERE`
* `YOUR_PATH_HERE`
* `YOUR_NUSCENES_PATH_HERE`
* `YOUR_TMPDIR_PATH_HERE`

For batch replacement (recommended for all except `YOUR_PATH_HERE`):

```bash
cd /mmdet/mmdetection3d/projects
find . -type f -exec grep -lF "YOUR_WORKDIR_PATH_HERE" {} + | xargs -r sed -i 's|YOUR_WORKDIR_PATH_HERE|/your/actual/workdir/path/here|g'

```

---

## Aggregation & Warm-Up Notes

* **FedDyn, FedMC, FedSelect, FedCKA rho setting:**
* Use a 10-round FedAvg warm-up phase to establish a stable initialization (all weights except the image backbone begin randomly initialized). This step significantly stabilizes optimization and boosts final metrics.

* **FedCKA tau setting:**
* Do **not** use the FedAvg warm-up phase; for layers are allowed to flow back if they align representationally. Using the 10 round warm-up deteriorates performance here, while for above descriped methods performance increases. It does use the first round FedAvg logic to initialize the same model for all clients.

* **FedCKA LocalSim setting:**
* Including LocalSim (Federated learning with partial model personalization, Pillutla et al., ICML 2022) for personalized subsets stabilizes and improves training. Replacing the config file and including the global model path is required for this. See FedSelect implementation for details.

---
## FedCKA Experiment Schedules

All FedCKA experiments use five clients, 40 communication rounds, one local training epoch per round, and 10 randomly selected CKA inputs per client. Percentage-based masks are cumulative, whereas threshold-based masks are re-evaluated every round and allow previously personalized layers to become globally shared again.

| Experiment         | Masking rule                                        | Initialization      | Masking schedule                                              | Final target | Mask persistence |
| ------------------ | --------------------------------------------------- | ------------------- | ------------------------------------------------------------- | -----------: | ---------------- |
| FedCKA $\rho=0.01$ | Personalize the lowest-CKA layers                   | FedAvg, rounds 1–10 | Add approximately 0.05% of parameters per round from round 11 |           1% | Cumulative       |
| FedCKA $\rho=0.10$ | Personalize the lowest-CKA layers                   | FedAvg, rounds 1–10 | Add approximately 0.5% of parameters per round from round 11  |          10% | Cumulative       |
| FedCKA $\rho=0.40$ | Personalize the lowest-CKA layers                   | FedAvg, rounds 1–10 | Add approximately 2% of parameters per round from round 11    |          40% | Cumulative       |
| FedCKA $\tau=0.05$ | Personalize layers with CKA similarity $s_l < 0.05$ | FedAvg, round 1     | Recompute the mask every round from round 2                   |      Dynamic | Elastic          |
| FedCKA $\tau=0.25$ | Personalize layers with CKA similarity $s_l < 0.25$ | FedAvg, round 1     | Recompute the mask every round from round 2                   |      Dynamic | Elastic          |
| FedCKA $\tau=0.50$ | Personalize layers with CKA similarity $s_l < 0.50$ | FedAvg, round 1     | Recompute the mask every round from round 2                   |      Dynamic | Elastic          |
| FedCKA $\tau=0.75$ | Personalize layers with CKA similarity $s_l < 0.75$ | FedAvg, round 1     | Recompute the mask every round from round 2                   |      Dynamic | Elastic          |

For the percentage-based experiments, the per-round increment equals $\rho/20$. Layers are selected in ascending order of CKA similarity until the incremental parameter budget is reached. Because complete layers are selected, the realized percentage can slightly exceed the nominal increment or final target. Once selected, a layer remains personalized for the rest of training.

For the threshold-based experiments, the first FedAvg round establishes a common fully initialized reference model. From round 2 onward, every layer is re-evaluated using its current CKA similarity. A layer is personalized while $s_l < \tau$ and returns to global aggregation when $s_l \geq \tau$.


---

## MMDetection3D runner settings

The experiment scripts support both MMDetection3D runner modes:

* `--slurm` submits distributed training through Slurm. Use this mode when you have access to a Slurm allocation and the required account, partition, and resource permissions.
* `--pytorch` launches distributed training directly with PyTorch. Use this mode when Slurm submission is unavailable or when running on a local machine or an already allocated node.

Choose exactly one mode for each run. The two modes should use equivalent GPU, CPU, memory, and time-limit settings so that results remain comparable. Check the selected `.sbatch` script and adapt its Slurm options, environment paths, and launcher arguments to your cluster before submitting the job.


---

## Seeds

Unless otherwise specified, the scripts use `seed=0`. For repeated runs, assign distinct sequential integer seeds (for example, `seed=1` for the second run and `seed=2` for the third) to support reproducibility and quantify run-to-run variability.

