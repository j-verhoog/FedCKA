#!/usr/bin/env python3
"""
NuScenes Robust Subset Creator (Metadata Filtering + Symlinking)
------------------------------------------------------------------
VERIFIED FOR MMDETECTION3D V1.x WITH TEMPORAL CONSISTENCY
------------------------------------------------------------------
Features:
1. Multi-configuration support (run multiple experiments in one go)
2. Temporal grouping (sequential vs interleaved client distribution)
3. Fair validation set handling (never dropped or subsampled)
4. Sample drop strategies (TAIL temporal vs RANDOM)
5. Fairness balancing (MIN-based, not MAX)
6. Data integrity (heals broken linked lists)
7. Robust symlinking (handles different mount points)

For each configuration:
   a. Separates train/val scenes from Excel.
   b. Applies fairness filtering to train samples only.
   c. Distributes train samples to clients (temporal or interleaved).
   d. Keeps 100% of validation samples for all clients.
   e. Traverses Scene -> Sample -> SampleData (including PREV sweeps).
   f. Filters JSON metadata tables to include ONLY assigned items.
   g. Heals broken sample_data pointers (prev/next).
   h. Writes clean JSONs to client_dir/v1.0-trainval.
   i. Symlinks only the relevant sensor files.
   j. Copies full maps (required for SDK).
"""

import os
import json
import shutil
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Tuple
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
import random
from concurrent.futures import ProcessPoolExecutor

# ================= GLOBAL CONFIGURATION =================
# Paths (same for all configurations)
NUSC_SOURCE_ROOT = Path("/YOUR_PATH_HERE/nuscenes")
EXCEL_PATH = Path("/YOUR_PATH_HERE/mmdet/mmdetection3d/projects/subsets_creation/scene_domains_summary.xlsx")
BASE_OUT_ROOT = Path("/YOUR_PATH_HERE/datasets/cmt_subsets")

# Global options
COPY_METHOD = "symlink"  # not used, but symlinks now by default
OVERWRITE = True  # Overwrite existing files in output
DEFAULT_RANDOM_SEED = 2026 # does not matter for main paper method, as no scenes are randomly dropped.

# Domain mapping
DOMAIN_MAPPING = {
    ("Boston", "day_rain"): "boston_day_rain",
    ("Boston", "day_clear"): "boston_day_clear",
    ("Singapore", "day_clear"): "singapore_day_clear",
    ("Singapore", "night_clear"): "singapore_night_clear",
    ("Singapore", "night_rain"): "singapore_night_rain",
}

# ================= EXPERIMENT CONFIGURATIONS =================
# Each configuration creates a different dataset variant
# These can be run sequentially to generate multiple experimental conditions
# takes approx 1.5hrs per run on 16-core CPU with SSD storage
CONFIGURATIONS = [
    {
        "name": "Default_NoFair_SingleClient",
        "temporal_grouping": False,  # False = Interleaved (stride), True = Sequential (chunks)
        "drop_strategy": "RANDOM",   # 'TAIL' = take first N, 'RANDOM' = random sample
        "fairness_mode": False,      # False, 'TOTAL', 'COMPARATIVE'
        "clients_per_domain": 1,     # Number of client subsets per domain
        "random_seed": DEFAULT_RANDOM_SEED,
    },
    # {
    #     "name": "Exp1_CompFair_SingleClient",
    #     "temporal_grouping": False,
    #     "drop_strategy": "RANDOM",
    #     "fairness_mode": "COMPARATIVE",
    #     "clients_per_domain": 1,
    #     "random_seed": DEFAULT_RANDOM_SEED,
    # },
    # {
    #     "name": "Exp2_TotalFair_SingleClient",
    #     "temporal_grouping": False,
    #     "drop_strategy": "RANDOM",
    #     "fairness_mode": "TOTAL",
    #     "clients_per_domain": 1,
    #     "random_seed": DEFAULT_RANDOM_SEED,
    # },
    # {
    #     "name": "Exp3_CompFair_DualClient",
    #     "temporal_grouping": False,
    #     "drop_strategy": "RANDOM",
    #     "fairness_mode": "COMPARATIVE",
    #     "clients_per_domain": 2,
    #     "random_seed": DEFAULT_RANDOM_SEED,
    # },
    # {
    #     "name": "Exp4_TotalFair_DualClient",
    #     "temporal_grouping": False,
    #     "drop_strategy": "RANDOM",
    #     "fairness_mode": "TOTAL",
    #     "clients_per_domain": 2,
    #     "random_seed": DEFAULT_RANDOM_SEED,
    # },
]

# Global dictionary to track excluded scenes per configuration per domain
# Structure: { config_name -> { domain -> [scene_tokens] } }
EXCLUDED_SCENES_REGISTRY = {}


# =================================================
# LOGGING SETUP (Dual Logger System)
# =================================================
BASE_OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Main logger (summary of important events)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(BASE_OUT_ROOT / "subset_creation.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Detailed logger (extensive debugging information)
detailed_logger = logging.getLogger('detailed')
detailed_logger.setLevel(logging.CRITICAL)                         # 'DEBUG' or 'CRITICAL' to surpress most logs
detailed_handler = logging.FileHandler(BASE_OUT_ROOT / "subset_creation_detailed.log", mode='w')
detailed_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(funcName)s:%(lineno)d: %(message)s"
))
detailed_logger.addHandler(detailed_handler)


logger.info("""
--- Complete nuScenes sweeps/samples check (anount of sweeps before a keyframe) ---
--- Results ---
Samples processed: 34149
Targeting: LIDAR_TOP
Samples missing target: 0
Samples with at least 1 LIDAR_TOP: 34149

Histogram (Prev Frames):
  0    : 850
  1    : 0
  2    : 0
  3    : 0
  4    : 0
  5    : 0
  6    : 0
  7    : 1045
  8    : 2112
  9    : 28360
  10   : 1316
  11   : 466
  12   : 0
  13   : 0
  14   : 0
  15   : 0
  16   : 0
  17   : 0
  18   : 0
  19   : 0
  20+  : 0
""")

# ================= HELPER FUNCTIONS =================

def load_scene_map(file_path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Reads the Excel and returns two mappings:
    - scene_token -> subset_name (e.g., 'boston_day_rain')
    - scene_token -> split (e.g., 'train' or 'val')
    
    Excel must have columns: 'scene_token', 'city', 'combo', 'split'
    """
    logger.info(f"Loading scene map from {file_path}")
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        df = pd.read_excel(file_path, engine='openpyxl')
    except Exception as e:
        logger.error("Failed to read Excel. Is openpyxl installed? (pip install openpyxl)")
        raise e

    # Normalize columns (strip whitespace)
    df.columns = [str(c).strip() for c in df.columns]
    
    # Check for required columns
    required = {'scene_token', 'city', 'combo', 'split'}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Excel missing required columns {required}. Found: {df.columns.tolist()}")

    scene_to_subset = {}
    scene_to_split = {}
    stats = {k: {"train": 0, "val": 0} for k in DOMAIN_MAPPING.values()}

    for _, row in df.iterrows():
        key = (row['city'], row['combo'])
        subset = DOMAIN_MAPPING.get(key)
        if not subset:
            continue
            
        scene_token = row['scene_token']
        split_val = str(row['split']).strip().lower()  # 'train' or 'val'
        
        if split_val not in ('train', 'val'):
            logger.warning(f"Scene {scene_token}: invalid split '{split_val}', defaulting to 'train'")
            split_val = 'train'
        
        scene_to_subset[scene_token] = subset
        scene_to_split[scene_token] = split_val
        stats[subset][split_val] += 1
            
    logger.info(f"Scene distribution: {json.dumps(stats, indent=2)}")
    return scene_to_subset, scene_to_split

def load_json_table(root: Path, table_name: str) -> List[Dict]:
    p = root / "v1.0-trainval" / f"{table_name}.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing metadata table: {p}")
    with open(p, 'r') as f:
        return json.load(f)

def save_json_table(data: List[Dict], root: Path, table_name: str):
    out_dir = root / "v1.0-trainval"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{table_name}.json", 'w') as f:
        json.dump(data, f, indent=0)

# ================= SYMBOLIC LINKING & UTILITY FUNCTIONS =================

def parallel_symlink_worker(task: Tuple[Path, Path]) -> bool:
    """
    Worker function for parallel symlink creation.
    Takes (src, dst) and creates a symlink with absolute path.
    
    NOTE: No logging in worker processes to avoid file lock contention.
    Worker processes cannot safely share file handles.
    
    Returns True if symlink created successfully, False otherwise.
    """
    src, dst = task
    
    # Check if source exists
    if not src.exists():
        return False
    
    # Use absolute path for the symlink target
    abs_src = src.resolve()
    
    # Remove existing destination if it exists
    try:
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink():
                dst.unlink()
            elif dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
    except Exception:
        return False
    
    # Create parent directories
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        return False
    
    # Create the symlink
    try:
        os.symlink(abs_src, dst)
        return True
    except Exception:
        return False

def safe_symlink(src: Path, dst: Path):
    """
    Creates a symlink, preferring relative paths for portability.
    Falls back to absolute symlinks if paths are on different mount points.
    Logs warnings for missing source files.
    
    CRITICAL FIX: Properly handles symlinks to directories by checking
    is_symlink() BEFORE is_dir(), since a symlink to a directory returns
    True for is_dir() but cannot be removed with rmtree().
    """
    if not src.exists():
        logger.warning(f"SOURCE FILE MISSING: {src} (cannot create symlink to {dst})")
        detailed_logger.debug(f"Missing source file: {src}")
        return  # Skip missing files

    if dst.exists() or dst.is_symlink():
        if not OVERWRITE:
            return
        try:
            # CRITICAL: Check is_symlink() BEFORE is_dir()
            # If dst is a symlink to a directory, is_dir() returns True
            # but rmtree() will fail on it
            if dst.is_symlink():
                # It's a symlink, remove it regardless of target type
                dst.unlink()
                detailed_logger.debug(f"Removed symlink: {dst}")
            elif dst.is_dir():
                # It's an actual directory, use rmtree
                shutil.rmtree(dst)
                detailed_logger.debug(f"Removed directory: {dst}")
            else:
                # It's a file, unlink it
                dst.unlink()
                detailed_logger.debug(f"Removed file: {dst}")
        except OSError as e:
            logger.warning(f"Failed to remove existing {dst}: {e}")
            detailed_logger.debug(f"OSError removing {dst}: {e}")
    
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Try relative path first (most portable)
        rel_src = os.path.relpath(src, dst.parent)
        os.symlink(src.resolve(), dst)
        detailed_logger.debug(f"Symlink created (absolute): {src.resolve()} -> {dst}")
    except ValueError:
        # ValueError: paths on different drives/mount points
        # Fallback to absolute symlink
        try:
            os.symlink(src.resolve(), dst)
            detailed_logger.debug(f"Symlink created (absolute): {src.resolve()} -> {dst}")
            logger.debug(f"Used absolute symlink for {src} (different mount point)")
        except Exception as e:
            logger.error(f"SYMLINK FAILED: {src} -> {dst}: {e}")
            detailed_logger.error(f"Symlink exception: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"SYMLINK FAILED: {src} -> {dst}: {e}")
        detailed_logger.error(f"Symlink exception: {e}", exc_info=True)

# ================= FAIRNESS & CLIENT DISTRIBUTION FUNCTIONS =================

def get_fairness_filtered_scenes(
    subset_name: str,
    scene_tokens: List[str],
    nusc: NuScenes,
    fairness_mode: str = False,
    drop_strategy: str = "TAIL",
    random_seed: int = DEFAULT_RANDOM_SEED
) -> Tuple[List[str], int, List[str]]:
    """
    Identifies which training scenes to include based on fairness mode.
    Works at SCENE level - entire scenes are kept or dropped together.
    Uses MIN-based capping to ensure fair comparison between domains.
    
    Args:
        subset_name: Domain name (e.g., 'boston_day_rain')
        scene_tokens: List of scene tokens for this domain
        nusc: NuScenes instance
        fairness_mode: False (no filtering), 'TOTAL', or 'COMPARATIVE'
        drop_strategy: 'TAIL' (keep first N) or 'RANDOM' (random sample)
        random_seed: Seed for reproducible random sampling
    
    Returns:
        Tuple of (kept_scene_tokens, limit_used, excluded_scene_tokens)
        - kept_scene_tokens: List of scene tokens to keep (ordered)
        - limit_used: The fairness limit that was applied (in scenes)
        - excluded_scene_tokens: List of excluded scene tokens
    
    Workflow:
        1. Collect ALL valid scenes for this domain
        2. Calculate fairness limit in SCENES (if fairness_mode enabled)
        3. Apply drop_strategy to select which scenes to keep
        4. Return filtered list and excluded scenes
    """
    detailed_logger.info(f"get_fairness_filtered_scenes: {subset_name}, fairness={fairness_mode}, drop_strategy={drop_strategy}")
    
    # ===== STEP 1: COLLECT ALL VALID SCENES =====
    all_valid_scenes = []
    
    for scene_token in scene_tokens:
        try:
            scene = nusc.get('scene', scene_token)
            all_valid_scenes.append(scene_token)
        except KeyError:
            logger.warning(f"Scene {scene_token} not found in DB")
            detailed_logger.debug(f"Scene not found: {scene_token}")
            continue
    
    logger.info(f"Collected {len(all_valid_scenes)} scenes for {subset_name}")
    detailed_logger.debug(f"Total scenes collected: {len(all_valid_scenes)}")
    
    if not all_valid_scenes:
        logger.warning(f"No valid scenes found for {subset_name}")
        return [], 0, scene_tokens
    
    # ===== STEP 2: CALCULATE FAIRNESS LIMIT (IN SCENES) =====
    if not fairness_mode:
        limit = len(all_valid_scenes)
        logger.debug(f"{subset_name}: No fairness, using all {limit} scenes")
    else:
        SINGAPORE_DOMAINS = {'singapore_day_clear', 'singapore_night_clear'}
        BOSTON_DOMAINS = {'boston_day_clear', 'boston_day_rain'}
        ALL_COMPARISON_DOMAINS = SINGAPORE_DOMAINS | BOSTON_DOMAINS
        
        if subset_name not in ALL_COMPARISON_DOMAINS:
            logger.warning(f"Domain {subset_name} not in comparison list. No fairness applied.")
            detailed_logger.debug(f"Domain {subset_name} not comparable")
            limit = len(all_valid_scenes)
        else:
            # Count TRAIN scenes only for fairness calculation
            domain_scene_counts = {}
            scene_map = load_scene_map(EXCEL_PATH)[0]  # scene_token -> domain
            scene_split = load_scene_map(EXCEL_PATH)[1]  # scene_token -> 'train'/'val'
            domain_scenes = {d: [] for d in ALL_COMPARISON_DOMAINS}
            
            # Collect only TRAIN scenes for each domain (exclude val)
            for token, domain in scene_map.items():
                if domain in ALL_COMPARISON_DOMAINS:
                    # Only include if this scene is marked as 'train' in Excel
                    if scene_split.get(token, '').lower() == 'train':
                        if token not in domain_scenes[domain]:
                            domain_scenes[domain].append(token)
            
            detailed_logger.debug(f"Counting TRAIN-ONLY scenes per domain (excluding val scenes)")
            
            # Count TRAIN scenes only
            for domain, tokens in domain_scenes.items():
                domain_scene_counts[domain] = len(tokens)
            
            detailed_logger.debug(f"TRAIN-ONLY scene counts per domain: {domain_scene_counts}")
            
            # Determine limit using MIN
            if fairness_mode == 'TOTAL':
                limit = min(domain_scene_counts.values()) if domain_scene_counts else len(all_valid_scenes)
                logger.info(f"FAIRNESS=TOTAL: {subset_name} limited to {limit} scenes (min across all)")
            
            elif fairness_mode == 'COMPARATIVE':
                singapore_limit = min(
                    domain_scene_counts.get('singapore_day_clear', float('inf')),
                    domain_scene_counts.get('singapore_night_clear', float('inf'))
                )
                boston_limit = min(
                    domain_scene_counts.get('boston_day_clear', float('inf')),
                    domain_scene_counts.get('boston_day_rain', float('inf'))
                )
                
                if singapore_limit == float('inf'):
                    singapore_limit = len(all_valid_scenes)
                if boston_limit == float('inf'):
                    boston_limit = len(all_valid_scenes)
                
                limit = singapore_limit if subset_name in SINGAPORE_DOMAINS else boston_limit
                logger.info(f"FAIRNESS=COMPARATIVE: {subset_name} limited to {limit} scenes")
            
            else:
                raise ValueError(f"Invalid fairness_mode: {fairness_mode}")
    
    # ===== STEP 3: APPLY DROP STRATEGY (at scene level) =====
    if len(all_valid_scenes) <= limit:
        keep_scenes = all_valid_scenes
        excluded_scenes = []
        logger.debug(f"{subset_name}: No reduction needed ({len(all_valid_scenes)} <= {limit})")
    else:
        logger.info(f"Fairness: Reducing {len(all_valid_scenes)} -> {limit} scenes using {drop_strategy}")
        detailed_logger.info(f"Applying drop_strategy={drop_strategy} to {subset_name}")
        
        if drop_strategy == "TAIL":
            # Keep first N scenes (temporal order preserved)
            keep_scenes = all_valid_scenes[:limit]
            excluded_scenes = all_valid_scenes[limit:]
            logger.debug(f"DROP=TAIL: Keeping first {limit} scenes (excluding last {len(excluded_scenes)})")
            detailed_logger.debug(f"TAIL: kept scenes [0:{limit}]")
        
        elif drop_strategy == "RANDOM":
            # Randomly select N scenes from entire set
            random.seed(random_seed)
            selected_indices = sorted(random.sample(range(len(all_valid_scenes)), limit))
            keep_scenes = [all_valid_scenes[i] for i in selected_indices]
            excluded_indices = set(range(len(all_valid_scenes))) - set(selected_indices)
            excluded_scenes = [all_valid_scenes[i] for i in sorted(excluded_indices)]
            logger.debug(f"DROP=RANDOM: Randomly selected {limit} scenes (excluding {len(excluded_scenes)}, seed={random_seed})")
            detailed_logger.debug(f"RANDOM: selected {len(selected_indices)} indices using seed {random_seed}")
        
        else:
            logger.warning(f"Unknown drop_strategy '{drop_strategy}', defaulting to TAIL")
            keep_scenes = all_valid_scenes[:limit]
            excluded_scenes = all_valid_scenes[limit:]
    
    detailed_logger.info(f"Final for {subset_name}: {len(keep_scenes)} scenes, {len(excluded_scenes)} excluded")
    return keep_scenes, limit, excluded_scenes




def distribute_scenes_to_clients(
    scene_tokens: List[str],
    num_clients: int,
    temporal_grouping: bool = False,
    drop_strategy: str = "TAIL",
    random_seed: int = DEFAULT_RANDOM_SEED
) -> Dict[int, List[str]]:
    """
    Distributes training scenes across multiple clients.
    CRITICAL: Complete scenes go to single clients - no mixing.
    
    Args:
        scene_tokens: List of scene tokens (should be ordered by time if temporal_grouping=True)
        num_clients: Number of clients
        temporal_grouping: True = sequential chunks, False = interleaved stride
        drop_strategy: 'TAIL' = take first N, 'RANDOM' = random sample
        random_seed: Seed for reproducible shuffling
    
    Returns:
        Dict[client_id (1-indexed) -> List[scene_tokens]]
    
    Logic:
        TEMPORAL_GROUPING=True (Sequential chunks):
            Client 1: [0:N]
            Client 2: [N:2N]
            Client 3: [2N:3N]
            Preserves temporal/spatial continuity per client
        
        TEMPORAL_GROUPING=False (Interleaved stride):
            Client 1: [0::num_clients]
            Client 2: [1::num_clients]
            Maximizes diversity per client, breaks continuity
        
        DROP_STRATEGY='TAIL': Take scenes sequentially from start
        DROP_STRATEGY='RANDOM': Randomly select scenes first, then organize
    """
    if num_clients < 1:
        raise ValueError(f"num_clients must be >= 1, got {num_clients}")
    
    if not scene_tokens:
        raise ValueError("Cannot distribute empty scene list")
    
    logger.debug(f"Distributing {len(scene_tokens)} scenes to {num_clients} clients "
                 f"(temporal={temporal_grouping}, drop_strategy={drop_strategy})")
    
    # Handle drop strategy
    if drop_strategy == "RANDOM":
        # Randomly select scenes
        random.seed(random_seed)
        shuffled = scene_tokens.copy()
        random.shuffle(shuffled)
        working_scenes = shuffled
    elif drop_strategy == "TAIL":
        # Use scenes sequentially (temporal order preserved)
        working_scenes = scene_tokens
    else:
        raise ValueError(f"Invalid drop_strategy: {drop_strategy}. Must be 'TAIL' or 'RANDOM'")
    
    client_scenes = {}
    
    if temporal_grouping:
        # Sequential chunking: divide into contiguous blocks
        chunk_size = len(working_scenes) // num_clients
        remainder = len(working_scenes) % num_clients
        
        idx = 0
        for client_id in range(1, num_clients + 1):
            # Give remainder scenes to the first clients
            size = chunk_size + (1 if client_id <= remainder else 0)
            client_scenes[client_id] = working_scenes[idx:idx + size]
            logger.debug(f"Client {client_id}: {size} scenes (temporal chunk)")
            idx += size
    
    else:
        # Interleaved stride: distribute round-robin
        for client_id in range(1, num_clients + 1):
            offset = client_id - 1  # 0-indexed offset
            scenes = working_scenes[offset::num_clients]
            client_scenes[client_id] = scenes
            logger.debug(f"Client {client_id}: {len(scenes)} scenes (interleaved)")
    
    # Verify distribution
    total_assigned = sum(len(v) for v in client_scenes.values())
    if total_assigned != len(working_scenes):
        raise RuntimeError(
            f"Distribution error: assigned {total_assigned} but had {len(working_scenes)}"
        )
    
    logger.info(f"Successfully distributed scenes: {[(i, len(v)) for i, v in sorted(client_scenes.items())]}")
    return client_scenes


# ================= DATA PROCESSING =================

def heal_pointers(records: List[Dict], table_name: str) -> None:
    """
    Fixes broken prev/next pointers in any table (sample_data, sample, etc.).
    
    When subsampling, some pointers may reference removed records.
    Set them to empty string to prevent crashes in mmdetection3d and external tools.
    
    CRITICAL FIX: Now heals both sample_data and sample pointers.
    External tools (NuScenes SDK utilities, some pipelines) may traverse samples,
    so sample.prev/next must also be healed for full data integrity.
    
    Args:
        records: List of record dicts with 'token', 'prev', 'next' keys
        table_name: Name of table (for logging, e.g., 'sample_data', 'sample')
    """
    valid_tokens = {r['token'] for r in records}
    healed_count = 0
    prev_healed = 0
    next_healed = 0
    
    for record in records:
        token = record['token']
        
        # Check and heal PREV pointer
        if record.get('prev'):
            if record['prev'] not in valid_tokens:
                detailed_logger.warning(f"Healing broken PREV in {table_name}: {token[:8]} -> {record['prev'][:8]} (missing)")
                record['prev'] = ""
                healed_count += 1
                prev_healed += 1
        
        # Check and heal NEXT pointer
        if record.get('next'):
            if record['next'] not in valid_tokens:
                detailed_logger.warning(f"Healing broken NEXT in {table_name}: {token[:8]} -> {record['next'][:8]} (missing)")
                record['next'] = ""
                healed_count += 1
                next_healed += 1
    
    if healed_count > 0:
        logger.warning(f"Healed {healed_count} broken pointers in {table_name} ({prev_healed} PREV, {next_healed} NEXT)")
        detailed_logger.info(f"Healing summary ({table_name}): {healed_count} total ({prev_healed} PREV, {next_healed} NEXT)")
    else:
        detailed_logger.debug(f"All {table_name} pointers valid (no healing needed)")



def _process_single_client_subset(
    subset_name: str,
    client_id: int,
    scene_tokens: List[str],
    client_train_scenes: Set[str],
    client_val_scenes: Set[str],
    subset_root: Path,
    nusc: NuScenes,
    raw_tables: Dict[str, List[Dict]]
):
    """
    Processes a single client's subset by filtering and linking data.
    NOW WORKS AT SCENE LEVEL: Derives samples from assigned scenes only.
    
    Each client gets:
    - Training scenes: partial (divided among clients)
    - Validation scenes: FULL (same for all clients in the domain)
    - All samples from these scenes are extracted
    
    Args:
        subset_name: Domain name (e.g., 'boston_day_rain')
        client_id: Client identifier (1-indexed)
        scene_tokens: All valid scene tokens for this domain
        client_train_scenes: Scene tokens assigned to this client for training
        client_val_scenes: ALL val scene tokens (full set for this domain)
        subset_root: Output directory for this client
        nusc: NuScenes instance
        raw_tables: Raw metadata tables
    
    Raises:
        KeyError: If referenced tokens are missing from the database
        RuntimeError: If filtering/writing fails
    """
    logger.debug(f"_process_single_client_subset: {subset_name}/client {client_id}")
    logger.debug(f"  Train scenes: {len(client_train_scenes)}, Val scenes: {len(client_val_scenes)}")
    
    if subset_root.exists() and OVERWRITE:
        logger.info(f"Cleaning up existing directory: {subset_root}")
        shutil.rmtree(subset_root)
    subset_root.mkdir(parents=True, exist_ok=True)

    # --- A. SCENE-TO-SAMPLE EXTRACTION ---
    # Extract ALL samples from the assigned scenes (no subsampling at sample level)
    all_client_scenes = client_train_scenes | client_val_scenes
    keep_samples = set()
    
    logger.debug(f"Extracting samples from {len(all_client_scenes)} assigned scenes...")
    
    for scene_token in tqdm(
        sorted(list(all_client_scenes)),
        desc=f"Extracting samples from scenes ({subset_name}/client{client_id})",
        leave=False
    ):
        try:
            scene = nusc.get('scene', scene_token)
        except KeyError:
            logger.error(f"Scene {scene_token} assigned to client but not in DB")
            continue
        
        # Traverse all samples in this scene
        curr_sample_token = scene['first_sample_token']
        sample_count = 0
        while curr_sample_token:
            keep_samples.add(curr_sample_token)
            sample_count += 1
            try:
                sample = nusc.get('sample', curr_sample_token)
                curr_sample_token = sample['next']
            except KeyError as e:
                logger.error(f"Sample chain broken at {curr_sample_token} in scene {scene_token}: {e}")
                break
        
        detailed_logger.debug(f"Scene {scene_token[:8]}: {sample_count} samples")
    
    logger.debug(f"Total samples extracted from assigned scenes: {len(keep_samples)}")
    logger.info(f"  {subset_name}/Client {client_id}: {len(keep_samples)} keyframes (samples)")
    
    # --- B. IDENTIFICATION PHASE (Graph Traversal) ---
    # Traverse graph for all extracted samples to identify related data
    keep_sample_data = set()  # Images/Lidar (Keyframes AND Sweeps)
    keep_instances = set()
    
    logger.debug(f"Traversing graph for {len(keep_samples)} samples...")
    
    lidar_sweep_warnings = []
    missing_samples = []
    
    for sample_token in tqdm(
        keep_samples,
        desc=f"Analyzing graph ({subset_name}/client{client_id})",
        leave=False
    ):
        try:
            sample = nusc.get('sample', sample_token)
        except KeyError as e:
            logger.error(f"Sample {sample_token} extracted from scene but not in DB")
            missing_samples.append(sample_token)
            raise KeyError(f"Missing sample: {sample_token}") from e

        # Walk through Sample Data (Sensors)
        for sensor, sd_token in sample['data'].items():
            # 1. Keep the Keyframe
            keep_sample_data.add(sd_token)
            
            # 2. Walk BACKWARDS for Sweeps (Crucial for mmdet3d Lidar)
            try:
                sd = nusc.get('sample_data', sd_token)
            except KeyError as e:
                logger.error(f"SampleData {sd_token} missing from DB")
                raise KeyError(f"Missing sample_data: {sd_token}") from e
            
            curr_sd_prev = sd['prev']
            sweep_count = 0
            
            while curr_sd_prev:
                try:
                    prev_sd = nusc.get('sample_data', curr_sd_prev)
                except KeyError as e:
                    logger.error(f"Broken sweep chain at {curr_sd_prev} for sensor {sensor}")
                    detailed_logger.error(f"Sweep chain broken: sample={sample_token[:8]}, sensor={sensor}, prev={curr_sd_prev[:8]}")
                    break
                
                if prev_sd['is_key_frame']:
                    break  # Stop, we reached the previous keyframe
                
                keep_sample_data.add(curr_sd_prev)
                sweep_count += 1
                curr_sd_prev = prev_sd['prev']
            
            # Check for insufficient LiDAR sweeps
            if 'LIDAR' in sensor and sweep_count < 7:
                msg = f"Sample {sample_token[:8]}: {sensor} has only {sweep_count} sweeps (< 7)"
                lidar_sweep_warnings.append(msg)
                detailed_logger.warning(f"Low LiDAR sweeps: {msg}")
        
        # Collect Instance IDs from Annotations
        for ann_token in sample['anns']:
            try:
                ann = nusc.get('sample_annotation', ann_token)
            except KeyError as e:
                logger.error(f"SampleAnnotation {ann_token} missing from DB")
                raise KeyError(f"Missing annotation: {ann_token}") from e
            keep_instances.add(ann['instance_token'])

    logger.debug(f"Graph traversal complete: {len(keep_sample_data)} sample_data, {len(keep_instances)} instances")
    
    # Log traverse summary
    if lidar_sweep_warnings:
        logger.warning(f"Found {len(lidar_sweep_warnings)} samples with < 7 LiDAR sweeps")
        detailed_logger.warning(f"Low LiDAR sweep summary:\n" + "\n".join(lidar_sweep_warnings[:20]))
        if len(lidar_sweep_warnings) > 20:
            detailed_logger.warning(f"... and {len(lidar_sweep_warnings) - 20} more")
    
    if missing_samples:
        logger.error(f"Found {len(missing_samples)} missing samples in DB")
        detailed_logger.error(f"Missing samples: {missing_samples}")

    # --- C. FILTERING PHASE ---
    logger.debug(f"Filtering metadata tables...")
    
    target_scene_tokens = set(all_client_scenes)
    new_tables = {}
    
    # Filter primary tables based on the sets we built above
    new_tables['scene'] = [s for s in raw_tables['scene'] if s['token'] in target_scene_tokens]
    new_tables['sample'] = [s for s in raw_tables['sample'] if s['token'] in keep_samples]
    new_tables['sample_data'] = [sd for sd in raw_tables['sample_data'] if sd['token'] in keep_sample_data]
    new_tables['sample_annotation'] = [a for a in raw_tables['sample_annotation'] if a['sample_token'] in keep_samples]
    new_tables['instance'] = [i for i in raw_tables['instance'] if i['token'] in keep_instances]
    
    # Filter dependent tables
    keep_logs = set(s['log_token'] for s in new_tables['scene'])
    new_tables['log'] = [l for l in raw_tables['log'] if l['token'] in keep_logs]
    
    keep_ego = set(sd['ego_pose_token'] for sd in new_tables['sample_data'])
    new_tables['ego_pose'] = [ep for ep in raw_tables['ego_pose'] if ep['token'] in keep_ego]
    
    keep_cs = set(sd['calibrated_sensor_token'] for sd in new_tables['sample_data'])
    new_tables['calibrated_sensor'] = [cs for cs in raw_tables['calibrated_sensor'] if cs['token'] in keep_cs]
    
    # Keep static tables full
    for t in ["map", "sensor", "category", "attribute", "visibility"]:
        new_tables[t] = raw_tables[t]

    logger.debug(f"Filtered tables: scene={len(new_tables['scene'])}, sample={len(new_tables['sample'])}, "
                 f"sample_data={len(new_tables['sample_data'])}, annotation={len(new_tables['sample_annotation'])}")

    # --- D. HEAL DATA INTEGRITY ---
    # Fix broken prev/next pointers in both sample and sample_data tables
    heal_pointers(new_tables['sample'], 'sample')
    heal_pointers(new_tables['sample_data'], 'sample_data')

    # --- E. Writing Phase ---
    try:
        for table_name, data in new_tables.items():
            save_json_table(data, subset_root, table_name)
        logger.debug(f"Successfully wrote JSON tables to {subset_root}")
    except Exception as e:
        logger.error(f"Failed to write JSON tables: {e}")
        raise RuntimeError(f"JSON writing failed: {e}") from e

    # --- F. Symlinking Phase (PARALLEL) ---
    logger.debug(f"Symlinking sensor files using parallel execution...")
    
    # Prepare the list of tasks (source_path, destination_path)
    symlink_tasks = []
    for sd in new_tables['sample_data']:
        filename = sd['filename']  # e.g. samples/CAM_FRONT/xxx.jpg
        src_path = NUSC_SOURCE_ROOT / filename
        dst_path = subset_root / filename
        symlink_tasks.append((src_path, dst_path))
    
    # Execute in parallel using ProcessPoolExecutor with 32 workers
    with ProcessPoolExecutor(max_workers=32) as executor:
        results = list(tqdm(
            executor.map(parallel_symlink_worker, symlink_tasks, chunksize=100),
            total=len(symlink_tasks),
            desc="Linking Files (Parallel)",
            leave=False
        ))
    
    # Count successful symlinks
    symlink_count = sum(1 for r in results if r)
    logger.debug(f"Created {symlink_count}/{len(symlink_tasks)} symlinks successfully")
    
    # --- G. Map Copying Phase ---
    logger.debug(f"Copying maps...")
    src_maps = NUSC_SOURCE_ROOT / "maps"
    dst_maps = subset_root / "maps"
    if src_maps.exists() and (not dst_maps.exists() or OVERWRITE):
        try:
            shutil.copytree(src_maps, dst_maps, dirs_exist_ok=True)
            logger.debug(f"Maps copied successfully")
        except Exception as e:
            logger.error(f"Failed to copy maps: {e}")
            raise RuntimeError(f"Map copying failed: {e}") from e
    elif not src_maps.exists():
        logger.warning(f"Source maps not found at {src_maps}")
    else:
        logger.debug(f"Maps already exist, skipping (OVERWRITE={OVERWRITE})")






def count_files_in_subsets() -> None:
    """
    Counts and logs the number of files in each subset folder.
    Provides summary statistics for verification and data auditing.
    """
    logger.info("="*80)
    logger.info("FILE COUNT SUMMARY")
    logger.info("="*80)
    
    if not BASE_OUT_ROOT.exists():
        logger.warning(f"Output root {BASE_OUT_ROOT} does not exist")
        return
    
    # Iterate through all configurations
    for config in CONFIGURATIONS:
        config_root = BASE_OUT_ROOT / config['name']
        if not config_root.exists():
            logger.warning(f"Configuration folder not found: {config_root}")
            continue
        
        logger.info(f"\nConfiguration: {config['name']}")
        logger.info(f"  temporal_grouping={config['temporal_grouping']}, "
                   f"fairness_mode={config['fairness_mode']}, "
                   f"clients_per_domain={config['clients_per_domain']}")
        
        # Iterate through domains
        for domain_dir in sorted(config_root.iterdir()):
            if not domain_dir.is_dir():
                continue
            
            domain_name = domain_dir.name
            
            # Check if multi-client (has numeric subdirectories) or single-client
            client_dirs = [d for d in domain_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            
            if client_dirs:
                # Multi-client setup
                logger.info(f"  Domain: {domain_name} (multi-client)")
                for client_dir in sorted(client_dirs, key=lambda x: int(x.name)):
                    client_id = client_dir.name
                    file_count = sum(1 for _ in client_dir.rglob('*') if _.is_file())
                    logger.info(f"    Client {client_id}: {file_count} files")
            else:
                # Single-client setup (files directly in domain folder)
                file_count = sum(1 for _ in domain_dir.rglob('*') if _.is_file())
                logger.info(f"  Domain: {domain_name}: {file_count} files")
    
    logger.info("="*80)


def generate_excluded_scenes_report() -> None:
    """
    Generates a comprehensive report of excluded scenes per configuration per domain.
    Saves to CSV files for easy analysis.
    """
    logger.info("="*80)
    logger.info("EXCLUDED SCENES REPORT")
    logger.info("="*80)
    
    if not EXCLUDED_SCENES_REGISTRY:
        logger.warning("No exclusion data to report")
        return
    
    # Per configuration
    for config_name, domain_dict in EXCLUDED_SCENES_REGISTRY.items():
        logger.info(f"\nConfiguration: {config_name}")
        
        # Create CSV file for this configuration
        csv_path = BASE_OUT_ROOT / f"excluded_scenes_{config_name}.csv"
        
        rows = []
        total_excluded = 0
        
        for domain_name in sorted(domain_dict.keys()):
            excluded_scenes = domain_dict[domain_name]
            num_excluded = len(excluded_scenes)
            total_excluded += num_excluded
            
            logger.info(f"  {domain_name}: {num_excluded} scenes excluded")
            detailed_logger.info(f"  Excluded scenes for {domain_name}: {excluded_scenes}")
            
            # Add rows to CSV
            for scene_token in excluded_scenes:
                rows.append({
                    'configuration': config_name,
                    'domain': domain_name,
                    'scene_token': scene_token
                })
        
        # Write CSV
        try:
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
            logger.info(f"  Excluded scenes CSV saved: {csv_path}")
        except Exception as e:
            logger.error(f"Failed to write excluded scenes CSV for {config_name}: {e}")
        
        logger.info(f"  Total excluded in {config_name}: {total_excluded} scenes")
    
    # Summary across all configurations
    logger.info("\n" + "="*80)
    logger.info("EXCLUSION SUMMARY ACROSS CONFIGURATIONS")
    logger.info("="*80)
    
    for config_name in sorted(EXCLUDED_SCENES_REGISTRY.keys()):
        domain_dict = EXCLUDED_SCENES_REGISTRY[config_name]
        summary = {}
        for domain_name, excluded_scenes in domain_dict.items():
            summary[domain_name] = len(excluded_scenes)
        logger.info(f"{config_name}: {summary}")
    
    logger.info("="*80)


def main():
    """
    Main execution loop that processes multiple configurations.
    Each configuration generates a complete dataset variant.
    """
    logger.info("="*80)
    logger.info("Starting NuScenes Robust Subset Creator (Multi-Configuration Mode)")
    logger.info(f"Number of configurations to process: {len(CONFIGURATIONS)}")
    logger.info("="*80)
    
    # Load static data (NuScenes DB, metadata tables)
    logger.info("Loading NuScenes database and metadata tables...")
    scene_to_subset, scene_to_split = load_scene_map(EXCEL_PATH)
    
    logger.info("Initializing NuScenes DB...")
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(NUSC_SOURCE_ROOT), verbose=True)

    logger.info("Loading raw JSON tables...")
    raw_tables = {}
    for t in ["scene", "sample", "sample_data", "sample_annotation", "instance", 
              "log", "ego_pose", "calibrated_sensor"]:
        raw_tables[t] = load_json_table(NUSC_SOURCE_ROOT, t)
    for t in ["map", "sensor", "category", "attribute", "visibility"]:
        raw_tables[t] = load_json_table(NUSC_SOURCE_ROOT, t)

    # ===== LOOP THROUGH CONFIGURATIONS =====
    for config_idx, config in enumerate(CONFIGURATIONS, 1):
        logger.info("="*80)
        logger.info(f"CONFIGURATION {config_idx}/{len(CONFIGURATIONS)}: {config['name']}")
        logger.info(f"  temporal_grouping: {config['temporal_grouping']}")
        logger.info(f"  drop_strategy: {config['drop_strategy']}")
        logger.info(f"  fairness_mode: {config['fairness_mode']}")
        logger.info(f"  clients_per_domain: {config['clients_per_domain']}")
        logger.info("="*80)
        
        out_root = BASE_OUT_ROOT / config['name']
        out_root.mkdir(parents=True, exist_ok=True)
        
        try:
            _process_configuration(
                config=config,
                out_root=out_root,
                scene_to_subset=scene_to_subset,
                scene_to_split=scene_to_split,
                nusc=nusc,
                raw_tables=raw_tables
            )
        except Exception as e:
            logger.error(f"Configuration {config['name']} failed: {e}", exc_info=True)
            raise

    logger.info("="*80)
    logger.info("SUCCESS. All configurations processed.")
    logger.info(f"Log: {BASE_OUT_ROOT / 'subset_creation.log'}")
    logger.info("="*80)
    
    # Generate excluded scenes report
    generate_excluded_scenes_report()
    
    # Count and log files in each subset
    count_files_in_subsets()


def _process_configuration(
    config: Dict,
    out_root: Path,
    scene_to_subset: Dict[str, str],
    scene_to_split: Dict[str, str],
    nusc: NuScenes,
    raw_tables: Dict[str, List[Dict]]
):
    """
    Process one configuration.
    
    Workflow (at SCENE level):
    1. Separate scenes into train/val by split
    2. Apply fairness filtering to train scenes (entire scenes kept/dropped)
    3. Extract ALL val scenes (no filtering)
    4. Distribute train scenes to clients
    5. For each client: merge train + val scene lists, extract ALL samples from those scenes
    6. Process client data
    """
    logger.info(f"Processing configuration: {config['name']}")
    
    # Initialize excluded scenes tracker for this configuration
    config_excluded = {}
    EXCLUDED_SCENES_REGISTRY[config['name']] = config_excluded
    
    # === STEP 1: Separate train/val scenes ===
    domains_train = {}  # subset_name -> list of train scene tokens
    domains_val = {}    # subset_name -> list of val scene tokens
    
    for scene_token, subset_name in scene_to_subset.items():
        split = scene_to_split.get(scene_token, 'train')  # Default to train if missing
        
        if subset_name not in domains_train:
            domains_train[subset_name] = []
            domains_val[subset_name] = []
            config_excluded[subset_name] = []  # Initialize excluded scenes list
        
        if split == 'train':
            domains_train[subset_name].append(scene_token)
        else:  # 'val'
            domains_val[subset_name].append(scene_token)
    
    # Log distribution
    for subset_name in sorted(set(scene_to_subset.values())):
        n_train = len(domains_train.get(subset_name, []))
        n_val = len(domains_val.get(subset_name, []))
        logger.info(f"  {subset_name}: {n_train} train scenes, {n_val} val scenes")
    
    # === STEP 2: Process each domain ===
    for subset_name in sorted(set(scene_to_subset.values())):
        train_scene_tokens = domains_train.get(subset_name, [])
        val_scene_tokens = domains_val.get(subset_name, [])
        
        if not train_scene_tokens and not val_scene_tokens:
            logger.warning(f"Skipping empty domain: {subset_name}")
            continue
        
        logger.info(f"Processing domain: {subset_name} ({len(train_scene_tokens)} train, {len(val_scene_tokens)} val)")
        
        # === STEP 2A: Apply fairness filtering to TRAIN scenes ===
        if train_scene_tokens:
            try:
                kept_train_scenes, fairness_limit, excluded_train_scenes = get_fairness_filtered_scenes(
                    subset_name,
                    train_scene_tokens,
                    nusc,
                    fairness_mode=config['fairness_mode'],
                    drop_strategy=config['drop_strategy'],
                    random_seed=config['random_seed']
                )
                # Track excluded scenes
                config_excluded[subset_name] = excluded_train_scenes
            except Exception as e:
                logger.error(f"Fairness filtering failed for {subset_name}: {e}")
                detailed_logger.error(f"Fairness filtering exception: {e}", exc_info=True)
                raise
            
            logger.info(f"  Train scenes after filtering: {len(kept_train_scenes)} (excluded: {len(excluded_train_scenes)})")
        else:
            kept_train_scenes = []
            excluded_train_scenes = []
            logger.info(f"  No train scenes for {subset_name}")
        
        # === STEP 2B: FULL VAL scenes (no filtering, no subsampling) ===
        kept_val_scenes = val_scene_tokens.copy()
        logger.info(f"  Validation scenes (no filtering): {len(kept_val_scenes)}")
        
        # Log final scene counts per domain after fairness filtering
        total_kept_scenes = len(kept_train_scenes) + len(kept_val_scenes)
        logger.info(f"  FINAL: {subset_name} - {len(kept_train_scenes)} train + {len(kept_val_scenes)} val = {total_kept_scenes} total scenes")
        
        # === STEP 2C: Distribute TRAIN scenes to clients ===
        if kept_train_scenes:
            try:
                client_train_dist = distribute_scenes_to_clients(
                    kept_train_scenes,
                    num_clients=config['clients_per_domain'],
                    temporal_grouping=config['temporal_grouping'],
                    drop_strategy=config['drop_strategy'],
                    random_seed=config['random_seed']
                )
            except Exception as e:
                logger.error(f"Client distribution failed for {subset_name}: {e}")
                raise
        else:
            # No train scenes, but still create clients with only validation
            client_train_dist = {
                i: [] for i in range(1, config['clients_per_domain'] + 1)
            }
        
        # === STEP 2D: Process each client ===
        for client_id in range(1, config['clients_per_domain'] + 1):
            client_train_scenes = set(client_train_dist.get(client_id, []))
            client_val_scenes = set(kept_val_scenes)
            
            # Combine all scenes for this client
            client_all_scenes = client_train_scenes | client_val_scenes
            
            logger.info(f"  {subset_name} / Client {client_id}: {len(client_train_scenes)} train scenes, {len(client_val_scenes)} val scenes (total {len(client_all_scenes)})")
            
            # Determine output directory
            if config['clients_per_domain'] == 1:
                # Single client: no subdirectory
                subset_root = out_root / subset_name
            else:
                # Multiple clients: create subdirectories
                subset_root = out_root / subset_name / str(client_id)
            
            try:
                _process_single_client_subset(
                    subset_name=subset_name,
                    client_id=client_id,
                    scene_tokens=sorted(list(client_all_scenes)),  # All scenes for this client
                    client_train_scenes=client_train_scenes,
                    client_val_scenes=client_val_scenes,
                    subset_root=subset_root,
                    nusc=nusc,
                    raw_tables=raw_tables
                )
            except Exception as e:
                logger.error(f"Processing failed for {subset_name}/client {client_id}: {e}", exc_info=True)
                raise
    
    logger.info(f"Configuration {config['name']} completed successfully")
    logger.info(f"Excluded scenes summary saved to registry")


if __name__ == "__main__":
    main()