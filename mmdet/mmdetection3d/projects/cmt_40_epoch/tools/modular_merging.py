import argparse
import time
import torch
import os
import string
import math
import importlib.util
import random

def parse_args():
    parser = argparse.ArgumentParser(description='Merge N models via FedAvg/FedBN and preserve state')
    
    parser.add_argument('--inputs', nargs='+', required=True, help='Paths to input model checkpoints')
    parser.add_argument('--outputs', nargs='+', required=True, help='Paths to save the distinct merged models')
    
    # Optional method flag. Defaults to fedavg to preserve original behavior.
    parser.add_argument('--method', type=str, default='fedavg', 
                    choices=['fedavg', 'fedbn', 'fednorm', 'fedrep', 'fed_bn_and_per', 'fedmedian', 'feddyn', 
                             'fed_dyn_bn_and_per', 'fedselect', 'fedselect_elastic', 'fedselect_fullelastic',
                             'fedselect_cka', 'fedselect_cka_elastic', 'pcgrad', 'fedmc'],
                    help='Aggregation method. Default is fedavg.')
    parser.add_argument('--config', type=str, default=None, 
                        help='Path to the MMDet3D config file (e.g., cmt_voxel01_vov_1600x640_cbgs_syncbn.py). Required for FedBN.')
    parser.add_argument(
        '--select-ratio',
        type=float,
        default=0.05,
        help='FedSelect-only: fraction of total params selected per round (default: 0.05).')
    parser.add_argument(
        '--max-sparsity',
        type=float,
        default=0.4,
        help='FedSelect-only: max personalized parameter fraction (default: 0.4).')

    # --- FedCKA arguments ---
    parser.add_argument('--data-dir-A', type=str, default=None, help='FedSelect CKA: Path to dataset-A directory.')
    parser.add_argument('--data-dir-B', type=str, default=None, help='FedSelect CKA: Path to dataset-B directory.')
    parser.add_argument('--data-dir-C', type=str, default=None, help='FedSelect CKA: Path to dataset-C directory.')
    parser.add_argument('--data-dir-D', type=str, default=None, help='FedSelect CKA: Path to dataset-D directory.')
    parser.add_argument('--data-dir-E', type=str, default=None, help='FedSelect CKA: Path to dataset-E directory.')
    parser.add_argument('--modality', type=str, default='lidar_camera', choices=['lidar', 'camera', 'lidar_camera'], help='FedSelect CKA: Modality for CKA comparison.')
    parser.add_argument('--runner-path', type=str, default='/YOUR_HOMEDIR_PATH_HERE/mmdet/mmdetection3d/projects/cmt_40_epoch/fedselect/scripts_cmt_copy/runner.py', help='FedSelect CKA: Path to runner.py script.')
    parser.add_argument('--cka-samples', type=int, default=10, help='FedSelect CKA: Max samples to process for CKA.')
    # ------------------------------------

    # optimizer reset argument    
    parser.add_argument('--no-reset-optimizer', action='store_true', default=True, help='If set, does not zero out optimizer states in merged checkpoints to reset momentum.')

    # fedCKA overwirte for dynamic sparsity
    parser.add_argument('--dynamic-sparsity', type=float, default=0.0, help='FedCKA: Dynamic sparsity fraction for this round (default: 0.0 for no dynamic sparsity).')

    # FedMC specific arguments
    parser.add_argument('--fisher_paths', nargs='+', type=str, default=None, 
                    help='Paths to the saved Fisher Information tensors (Required ONLY for FedMC)')
    
    for i, char in enumerate(string.ascii_lowercase):
        help_text = f'Weight for model {char.upper()}' if i < 5 else argparse.SUPPRESS
        parser.add_argument(f'--weight-{char}', type=float, default=None, help=help_text)
        
    args = parser.parse_args()
    return args

def flatten_tensors(state_dict, valid_keys):
    """Flattens specified keys of a state_dict into a single 1D PyTorch tensor."""
    return torch.cat([state_dict[k].flatten() for k in valid_keys])

def unflatten_tensors(flat_tensor, reference_dict, valid_keys):
    """Restores a 1D tensor back into a state_dict dictionary structure."""
    unflattened = {}
    idx = 0
    for k in valid_keys:
        numel = reference_dict[k].numel()
        shape = reference_dict[k].shape
        unflattened[k] = flat_tensor[idx:idx+numel].view(shape).clone()
        idx += numel
    return unflattened


def reset_optimizer_state(ckpt, no_reset_optimizer=False):
    """Reset checkpoint optimizer tensors unless explicitly disabled."""
    if no_reset_optimizer:
        return

    if 'optimizer' in ckpt and 'state' in ckpt['optimizer']:
        for param_id in ckpt['optimizer']['state']:
            for key in ckpt['optimizer']['state'][param_id]:
                value = ckpt['optimizer']['state'][param_id][key]
                if torch.is_tensor(value):
                    value.zero_()


def PCGRAD(models, output_paths, norm_weights, client_ids, prev_global_path="/workspace/work_dirs/pcgrad_states/global_model.pth", no_reset_optimizer=False):
    """
    PCGRAD - With Configurable Exclusions. Naming is off, as FedOMG is more elaborate version of this, and this is simpler PCGrad.
    """
    # =====================================================================
    # [CONFIGURATION] EXCLUSION SETTINGS
    # =====================================================================
    # By default, we exclude BatchNorm running stats from gradient matching.
    # You can add any layer prefix or string here to exclude it.
    # Any parameter whose name contains any of these strings will bypass the 
    # projection math and simply be aggregated via standard FedAvg.

    # Using this projection logic on the BatchNorm running stats can lead to instability, so it is possible to exclude them with:
    # EXCLUDE_PREFIXES = ['bn','running_mean', 'running_var', 'num_batches_tracked']
    # =====================================================================
    
    EXCLUDE_PREFIXES = ['num_batches_tracked']
    KEEP_PRIVATE = True     # keeps the excluded prefixes strictly local and does not merge them into the client models

    print("\n" + "="*50)
    print("Running PCGRAD (On-Server Matching Gradient) Aggregation...")
    print("="*50)
    
    os.makedirs(os.path.dirname(prev_global_path), exist_ok=True)
    num_clients = len(models)

    # ---------------------------------------------------------
    # Round 0 check (Initialization fallback)
    # ---------------------------------------------------------
    if not os.path.exists(prev_global_path):
        print(f"[WARNING] No previous global model found at {prev_global_path}.")
        print("Initializing GradSplit baseline by running standard FedAvg for Round 0...")
        
        # Run FedAvg to create the initial global consensus
        fedavg(models, [prev_global_path] * num_clients, norm_weights) 
        
        # Load the newly created global model to get the averaged weights
        global_ckpt = torch.load(prev_global_path, map_location='cpu')
        global_state = global_ckpt['state_dict']
        
        for in_path, out_path in zip(models, output_paths):
            # Load the client's individual checkpoint (preserves their 'step', 'meta', etc.)
            client_ckpt = torch.load(in_path, map_location='cpu')
            
            # Safely overwrite ONLY the model weights with the global consensus
            for k, v in global_state.items():
                if k in client_ckpt['state_dict']:
                    client_ckpt['state_dict'][k] = v.clone()
            
            # Save the updated, individualized checkpoint for the client
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(client_ckpt, out_path)
            
        print("Round 0 initialization complete. Client step counts preserved.")
        return

    global_ckpt = torch.load(prev_global_path, map_location='cpu')
    global_state = global_ckpt['state_dict']
    
    # Strictly separate FedOMG keys from standard FedAvg keys based on your config
    fedomg_keys = []
    fedavg_keys = []
    
    for k, v in global_state.items():
        if v.is_floating_point():
            if any(excl in k for excl in EXCLUDE_PREFIXES):
                fedavg_keys.append(k)
            else:
                fedomg_keys.append(k)

    total_params = sum(global_state[k].numel() for k in fedomg_keys)
    print(f"PCGRAD: Conflict tracking on {len(fedomg_keys)} layers ({total_params:,} true weights).")
    if not KEEP_PRIVATE:
        print(f"PCGRAD: Standard averaging on {len(fedavg_keys)} excluded layers/trackers.")
    else:
        print(f"PCGRAD: Not including {len(fedavg_keys)} private layers.")

    # 1. Compute Pseudo-Gradients AND Accumulate FedAvg layers
    print("\n--- Phase 1: Computing Pseudo-Gradients ---")
    flat_global = flatten_tensors(global_state, fedomg_keys)
    
    client_pseudo_grads = []
    fedavg_accumulators = {k: torch.zeros_like(global_state[k]) for k in fedavg_keys}
    
    for i, (m_path, cid, w_i) in enumerate(zip(models, client_ids, norm_weights)):
        ckpt_i = torch.load(m_path, map_location='cpu')
        state_i = ckpt_i['state_dict']
        
        # Extract PCGRAD gradient
        flat_client = flatten_tensors(state_i, fedomg_keys)
        pseudo_grad = flat_global - flat_client
        client_pseudo_grads.append(pseudo_grad)
        
        grad_norm = torch.norm(pseudo_grad).item()
        print(f"  [{cid}] Extracted true weight gradient. L2 Norm: {grad_norm:.4f}")
        
        # Accumulate FedAvg layers
        for k in fedavg_keys:
            if k in state_i:
                fedavg_accumulators[k] += state_i[k] * w_i
                
        del ckpt_i, flat_client

    # 2. Conflict Resolution via Gradient Projection
    print("\n--- Phase 2: Resolving Domain Conflicts (Gradient Matching) ---")
    total_conflicts_resolved = 0
    
    for i in range(num_clients):
        check_order = list(range(num_clients))
        random.shuffle(check_order)
        
        for j in check_order:
            if i == j: 
                continue
            
            dot_product = torch.dot(client_pseudo_grads[i], client_pseudo_grads[j]).item()
            cos_sim = dot_product / (torch.norm(client_pseudo_grads[i]).item() * torch.norm(client_pseudo_grads[j]).item() + 1e-8)
            print(f"  [EVAL] {client_ids[i]} vs {client_ids[j]} -> Cosine Sim: {cos_sim:.4f}")
            
            if dot_product < 0:
                total_conflicts_resolved += 1
                norm_sq_j = torch.dot(client_pseudo_grads[j], client_pseudo_grads[j]).item() + 1e-8
                
                print(f"      [!] CONFLICT DETECTED. Projecting {client_ids[i]} away from {client_ids[j]}.")
                
                projection_scalar = dot_product / norm_sq_j
                client_pseudo_grads[i] = client_pseudo_grads[i] - (projection_scalar * client_pseudo_grads[j])

    print(f"\nPCGRAD Phase 2 Complete. Total pairwise conflicts resolved: {total_conflicts_resolved}")

    # 3. Aggregate Aligned Gradients
    print("\n--- Phase 3: Aggregating Aligned Gradients ---")
    aggregated_grad = torch.zeros_like(flat_global)
    for i, w_i in enumerate(norm_weights):
        aggregated_grad += client_pseudo_grads[i] * w_i
        
    final_grad_norm = torch.norm(aggregated_grad).item()
    print(f"  Aggregated Global Gradient L2 Norm: {final_grad_norm:.4f}")

    # 4. Update Global Model 
    new_flat_global = flat_global - aggregated_grad
    new_global_weights = unflatten_tensors(new_flat_global, global_state, fedomg_keys)
    
    # Merge both PCGRAD weights and FedAvg excluded weights back in
    for k in fedomg_keys:
        global_ckpt['state_dict'][k] = new_global_weights[k]
    for k in fedavg_keys:
        global_ckpt['state_dict'][k] = fedavg_accumulators[k]
        
    print(f"  Saving new conflict-free global model to {prev_global_path}")
    torch.save(global_ckpt, prev_global_path)

    # 5. Distribute
    print("\n--- Phase 4: Distributing Global Model ---")
    for in_path, out_path, cid in zip(models, output_paths, client_ids):
        ckpt = torch.load(in_path, map_location='cpu')
        
        for k in global_ckpt['state_dict'].keys():
            if not k in fedavg_keys:                # Only overwrite FedOMG keys, keep FedAvg keys local
                if k in ckpt['state_dict']:
                    ckpt['state_dict'][k] = global_ckpt['state_dict'][k].clone()
                
        reset_optimizer_state(ckpt, no_reset_optimizer)
                        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(ckpt, out_path)
        print(f"  Saved PCGRAD synchronized model for {cid} to {out_path}")
        
    print("="*50)
    print("PCGRAD Aggregation Complete.\n")

def fedavg(models, output_paths, norm_weights, no_reset_optimizer=False):
    """
    Original FedAvg implementation. Averages all valid floating-point keys.
    Maintains exact original functionality.
    """
    print("Running standard FedAvg...")
    
    # 1. Setup accumulators for the N-way average using the first model's structure
    temp_ckpt = torch.load(models[0], map_location='cpu')

    # 2. Setup accumulators for the N-way average
    running_sum = {}
    presence_weights = {}
    
    for k, v in temp_ckpt['state_dict'].items():
        if not v.is_floating_point() or 'num_batches_tracked' in k:
             continue
             
        running_sum[k] = torch.zeros_like(v)
        presence_weights[k] = 0.0
        
    del temp_ckpt
    
    # 3. Iterate sequentially through remaining models
    print("Averaging weights...")
    for i in range(len(models)):
        m_path = models[i]
        w_i = norm_weights[i]
        
        ckpt_i = torch.load(m_path, map_location='cpu')
        state_dict_i = ckpt_i['state_dict']
        
        for k in running_sum.keys():
            if k in state_dict_i:
                running_sum[k] += state_dict_i[k] * w_i
                presence_weights[k] += w_i
                
        # Free memory after processing each model
        del ckpt_i 
        
    # Calculate final averaged weights
    averaged_weights = {k: running_sum[k] / presence_weights[k] for k in running_sum.keys()}

    # 4. Inject Averaged Weights into EACH Model and Save
    for in_path, out_path in zip(models, output_paths):
        print(f"Updating and saving individual state for: {out_path}")
        ckpt = torch.load(in_path, map_location='cpu')
        
        for k in averaged_weights.keys():
            ckpt['state_dict'][k] = averaged_weights[k]
        
        reset_optimizer_state(ckpt, no_reset_optimizer)
                        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(ckpt, out_path)
        print(f"Saved merged model with optimizer state to {out_path}")

def fedbn(models, output_paths, norm_weights, model_instance, no_reset_optimizer=False):
    """
    FedBN implementation. Uses the provided PyTorch model instance to map out
    BatchNorm layers and excludes their weights, biases, and running stats from averaging.
    """
    print("Running FedBN. Extracting BatchNorm topology from model class...")
    
    # Identify all base names of BatchNorm layers using the actual PyTorch classes
    bn_prefixes = set()
    total_layers = 0
    for name, module in model_instance.named_modules():
        total_layers += 1
        # This catches nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, and SyncBatchNorm
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            clean_name = name.replace('module.', '')
            bn_prefixes.add(clean_name)
            
    print(f"Total layers in model: {total_layers}")
    print(f"Identified {len(bn_prefixes)} BatchNorm layers to keep local.")

    # 1. Setup accumulators using the first model's structure
    temp_ckpt = torch.load(models[0], map_location='cpu')

    running_sum = {}
    presence_weights = {}
    
    for k, v in temp_ckpt['state_dict'].items():
        if not v.is_floating_point() or 'num_batches_tracked' in k:
             continue
        
        clean_k = k.replace('module.', '')
             
        # Skip this key if it belongs to any identified BatchNorm layer
        if any(clean_k.startswith(prefix + '.') for prefix in bn_prefixes):
            continue
             
        running_sum[k] = torch.zeros_like(v)
        presence_weights[k] = 0.0
        
    del temp_ckpt
    
    # 2. Iterate sequentially through models
    print("Averaging non-BN weights...")
    for i in range(len(models)):
        m_path = models[i]
        w_i = norm_weights[i]
        
        ckpt_i = torch.load(m_path, map_location='cpu')
        state_dict_i = ckpt_i['state_dict']
        
        for k in running_sum.keys():
            if k in state_dict_i:
                running_sum[k] += state_dict_i[k] * w_i
                presence_weights[k] += w_i
                
        del ckpt_i 
        
    averaged_weights = {k: running_sum[k] / presence_weights[k] for k in running_sum.keys()}

    # 3. Inject Averaged Weights into EACH Model and Save
    for in_path, out_path in zip(models, output_paths):
        print(f"Updating and saving individual state for: {out_path}")
        ckpt = torch.load(in_path, map_location='cpu')
        
        # BN keys aren't in averaged_weights, so their local state remains untouched
        for k in averaged_weights.keys():
            ckpt['state_dict'][k] = averaged_weights[k]
            
        reset_optimizer_state(ckpt, no_reset_optimizer)
                        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(ckpt, out_path)
        print(f"Saved merged model with optimizer state to {out_path}")

def fedrep(models, output_paths, norm_weights, no_reset_optimizer=False):
    """
    fedrep implementation. Averages the backbone and neck, but keeps the 
    task head subsets completely local and personalized.
    """
    print("Running fedrep. Keeping task head local...")
    
    # 1. Setup accumulators using the first model's structure
    temp_ckpt = torch.load(models[0], map_location='cpu')

    running_sum = {}
    presence_weights = {}
    
    personalized_layers = 0
    global_layers = 0
    # Define specifically which sub-components of the head to keep local
    local_prefixes = (
        'pts_bbox_head.common_heads',
        'pts_bbox_head.separate_head',
        'pts_bbox_head.tasks',
        'pts_bbox_head.task_heads'       # <--- FIXED PREFIX
    )
    
    for k, v in temp_ckpt['state_dict'].items():
        if not v.is_floating_point() or 'num_batches_tracked' in k:
             continue
             
        # Only skip the specific task-prediction heads, not the whole transformer
        if any(k.startswith(prefix) for prefix in local_prefixes):
            personalized_layers += 1
            continue
             
        running_sum[k] = torch.zeros_like(v)
        presence_weights[k] = 0.0
        global_layers += 1
        
    del temp_ckpt
    
    print(f"Identified {personalized_layers} personalized head layers to keep local.")
    print(f"Identified {global_layers} global layers to average across models.")
    
    # 2. Iterate sequentially through models
    print("Averaging backbone and neck weights...")
    for i in range(len(models)):
        m_path = models[i]
        w_i = norm_weights[i]
        
        ckpt_i = torch.load(m_path, map_location='cpu')
        state_dict_i = ckpt_i['state_dict']
        
        for k in running_sum.keys():
            if k in state_dict_i:
                running_sum[k] += state_dict_i[k] * w_i
                presence_weights[k] += w_i
                
        del ckpt_i 
        
    averaged_weights = {k: running_sum[k] / presence_weights[k] for k in running_sum.keys()}

    # 3. Inject Averaged Weights and Save
    for in_path, out_path in zip(models, output_paths):
        print(f"Updating and saving individual state for: {out_path}")
        ckpt = torch.load(in_path, map_location='cpu')
        
        for k in averaged_weights.keys():
            ckpt['state_dict'][k] = averaged_weights[k]
            
        reset_optimizer_state(ckpt, no_reset_optimizer)
                        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(ckpt, out_path)
        print(f"Saved merged model to {out_path}")

def feddyn(models, output_paths, norm_weights, alpha=0.01, work_dir="work_dirs/feddyn_states", no_reset_optimizer=False):
    print("Running FedDyn Aggregation...")

    if len(models) != len(output_paths):
        raise ValueError("The number of input and output model paths must match.")

    client_ids = ["ModelA", "ModelB", "ModelC", "ModelD", "ModelE"]
    if len(models) != len(client_ids):
        raise ValueError(
            f"FedDyn expects {len(client_ids)} clients, but received {len(models)}."
        )

    if len(norm_weights) != len(models):
        raise ValueError(
            "The number of aggregation weights must match the number of models."
        )

    if alpha <= 0:
        raise ValueError("FedDyn alpha must be greater than zero.")

    # Ensure that the aggregation weights form a normalized weighted mean.
    weight_sum = float(sum(norm_weights))
    if weight_sum <= 0:
        raise ValueError("FedDyn aggregation weights must have a positive sum.")

    normalized_weights = [
        float(weight) / weight_sum for weight in norm_weights
    ]

    # ------------------------------------------------------------------
    # 1. Compute the weighted average of the incoming client models.
    # ------------------------------------------------------------------
    temp_ckpt = torch.load(models[0], map_location="cpu")

    running_sum = {
        key: torch.zeros_like(value)
        for key, value in temp_ckpt["state_dict"].items()
        if value.is_floating_point() and "num_batches_tracked" not in key
    }

    del temp_ckpt

    for model_path, client_weight in zip(models, normalized_weights):
        client_ckpt = torch.load(model_path, map_location="cpu")
        client_state = client_ckpt["state_dict"]

        for key in running_sum:
            if key not in client_state:
                raise KeyError(
                    f"Parameter '{key}' is missing from checkpoint: {model_path}"
                )

            running_sum[key] += client_state[key] * client_weight

        del client_ckpt

    averaged_weights = {
        key: value
        for key, value in running_sum.items()
    }

    # ------------------------------------------------------------------
    # 2. Load the cumulative client h states.
    #
    # The client hook uses:
    #
    #     h_i <- h_i - alpha * (w_i - global)
    #
    # Therefore, with this sign convention, the FedDyn global model is:
    #
    #     global <- weighted_mean(w_i) - weighted_mean(h_i) / alpha
    #
    # No additional persistent server_h_state is required because the
    # individual client h states are already cumulative.
    # ------------------------------------------------------------------
    client_h_states = []

    for client_id in client_ids:
        client_h_path = os.path.join(
            work_dir,
            f"{client_id}_h_state.pth"
        )

        if not os.path.exists(client_h_path):
            raise FileNotFoundError(
                f"Missing FedDyn state for {client_id}: {client_h_path}"
            )

        client_h = torch.load(client_h_path, map_location="cpu")
        client_h_states.append(client_h)

    def get_client_h_value(client_h, parameter_key):
        """Handle checkpoints with or without a leading 'module.' prefix."""
        if parameter_key in client_h:
            return client_h[parameter_key]

        clean_key = parameter_key.removeprefix("module.")
        if clean_key in client_h:
            return client_h[clean_key]

        prefixed_key = f"module.{clean_key}"
        if prefixed_key in client_h:
            return client_h[prefixed_key]

        return None

    # ------------------------------------------------------------------
    # 3. Apply the FedDyn correction to the averaged model.
    # ------------------------------------------------------------------
    corrected_keys = 0
    untracked_keys = 0

    for key, averaged_value in averaged_weights.items():
        client_h_values = [
            get_client_h_value(client_h, key)
            for client_h in client_h_states
        ]

        available_count = sum(
            value is not None for value in client_h_values
        )

        if available_count == len(client_h_states):
            mean_h = torch.zeros_like(averaged_value)

            for client_weight, client_h_value in zip(
                normalized_weights,
                client_h_values
            ):
                if client_h_value.shape != averaged_value.shape:
                    raise ValueError(
                        f"Shape mismatch for FedDyn state '{key}': "
                        f"model shape={tuple(averaged_value.shape)}, "
                        f"h-state shape={tuple(client_h_value.shape)}"
                    )

                mean_h += client_h_value.to(
                    dtype=averaged_value.dtype
                ) * client_weight

            averaged_weights[key] = averaged_value - (mean_h / alpha)
            corrected_keys += 1

        elif available_count == 0:
            # State-dict buffers that are not trainable parameters are not
            # necessarily tracked by the client hook. Keep their FedAvg value.
            untracked_keys += 1

        else:
            raise KeyError(
                f"FedDyn state '{key}' is present for only "
                f"{available_count}/{len(client_h_states)} clients. "
                "Refusing to aggregate inconsistent client states."
            )

    print(
        f"FedDyn: Applied dynamic correction to {corrected_keys} keys."
    )
    print(
        f"FedDyn: Kept the FedAvg value for {untracked_keys} "
        "untracked state-dict keys."
    )

    # ------------------------------------------------------------------
    # 4. Inject the corrected global weights and save each client model.
    # ------------------------------------------------------------------
    for input_path, output_path in zip(models, output_paths):
        checkpoint = torch.load(input_path, map_location="cpu")

        for key, value in averaged_weights.items():
            checkpoint["state_dict"][key] = value

        reset_optimizer_state(
            checkpoint,
            no_reset_optimizer
        )

        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )
        torch.save(checkpoint, output_path)

        print(f"Saved FedDyn model to: {output_path}")

def fedselect(models, output_paths, norm_weights, client_ids, prev_global_path="/workspace/work_dirs/fedselect_states/global_model.pth", mask_dir="/workspace/work_dirs/fedselect_masks", select_ratio=0.05, max_sparsity=0.5, no_reset_optimizer=False):
    """
    FedSelect implementation (CVPR 2024). 
    Automatically discovers and freezes personalized subnetworks for each client 
    based on the magnitude of weight changes, then aggregates only the shared parameters.
    """
    print(f"Running FedSelect Aggregation (select_ratio={select_ratio}, max_sparsity={max_sparsity})...")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(os.path.dirname(prev_global_path), exist_ok=True)

    # # 1. Load Pre-Training Global Weights
    # if not os.path.exists(prev_global_path):
    #     print(f"No previous global model found at {prev_global_path}. Exiting FedSelect since we need a baseline for difference calculation. Please run one round of standard FedAvg first to initialize the global model.")
    #     exit(1)

    # 1. Load / Recover Pre-Training Global Weights
    if not os.path.exists(prev_global_path):
        import re

        print(
            f"No previous global model found at {prev_global_path}. "
            "Attempting reconstruction from previous-round merged client models..."
        )

        # ---------------------------------------------------------
        # Determine current round from output path.
        # Example:
        #   /workspace/work_dirs/round_11/merged_A.pth
        # -> current_round = 11
        # ---------------------------------------------------------
        round_match = re.search(r'/round_(\d+)(?:/|$)', output_paths[0])

        if round_match is None:
            print(
                "ERROR: Could not determine current round from output path:\n"
                f"  {output_paths[0]}"
            )
            print("Expected a path containing '/round_X/'.")
            exit(1)

        current_round = int(round_match.group(1))
        previous_round = current_round - 1

        if previous_round < 1:
            print(
                f"ERROR: Current round is {current_round}, so there is no "
                "previous round from which to reconstruct the global model."
            )
            exit(1)

        print(f"Current round: {current_round}")
        print(f"Recovering global model from round {previous_round}.")

        # ---------------------------------------------------------
        # Construct previous-round merged paths by replacing
        # /round_CURRENT/ with /round_PREVIOUS/.
        # ---------------------------------------------------------
        previous_merged_paths = []

        for output_path in output_paths:
            previous_path = output_path.replace(
                f"/round_{current_round}/",
                f"/round_{previous_round}/"
            )
            previous_merged_paths.append(previous_path)

        # ---------------------------------------------------------
        # Check that all previous merged models exist
        # ---------------------------------------------------------
        missing_merged = [
            path for path in previous_merged_paths
            if not os.path.exists(path)
        ]

        if missing_merged:
            print(
                f"ERROR: Cannot reconstruct global model. "
                f"Missing merged checkpoints from round {previous_round}:"
            )

            for path in missing_merged:
                print(f"  - {path}")

            print("Exiting FedSelect.")
            exit(1)

        print(
            f"Found all {len(previous_merged_paths)} merged client models "
            f"from round {previous_round}."
        )

        # ---------------------------------------------------------
        # Load previous merged models once
        # ---------------------------------------------------------
        previous_ckpts = [
            torch.load(path, map_location='cpu')
            for path in previous_merged_paths
        ]

        previous_states = [
            ckpt['state_dict']
            for ckpt in previous_ckpts
        ]

        # Use first checkpoint as template
        recovered_ckpt = previous_ckpts[0]
        recovered_state = recovered_ckpt['state_dict']

        valid_keys_recovery = [
            k for k, v in recovered_state.items()
            if v.is_floating_point()
            and 'num_batches_tracked' not in k
        ]

        # ---------------------------------------------------------
        # Check for cumulative FedSelect masks
        # ---------------------------------------------------------
        mask_paths = [
            os.path.join(mask_dir, f"{cid}_mask.pth")
            for cid in client_ids
        ]

        missing_masks = [
            path for path in mask_paths
            if not os.path.exists(path)
        ]

        # =========================================================
        # CASE 1:
        # No masks exist yet.
        #
        # This is expected when recovering the FedAvg global model
        # immediately before the first FedSelect round.
        #
        # All previous merged models must therefore be identical.
        # =========================================================
        if missing_masks:

            print(
                "FedSelect masks are not available. "
                "Checking whether previous-round merged models are identical..."
            )

            models_identical = True

            for k in valid_keys_recovery:
                reference = previous_states[0][k]

                for state_i in previous_states[1:]:
                    if (
                        k not in state_i
                        or not torch.equal(reference, state_i[k])
                    ):
                        models_identical = False
                        break

                if not models_identical:
                    break

            if not models_identical:
                print(
                    "ERROR: Previous merged models are not identical, "
                    "but the required FedSelect masks are missing."
                )

                print("Missing masks:")
                for path in missing_masks:
                    print(f"  - {path}")

                print(
                    "Cannot determine which parameters are global versus "
                    "personalized. Exiting FedSelect."
                )
                exit(1)

            # All merged models are identical -> exact global model
            print(
                f"All merged models from round {previous_round} are identical. "
                "Using merged_A as the exact global model."
            )

        # =========================================================
        # CASE 2:
        # Masks exist -> reconstruct global model parameter-wise
        # =========================================================
        else:

            print(
                "All required masks found. "
                "Reconstructing global parameters..."
            )

            client_masks = [
                torch.load(path, map_location='cpu')
                for path in mask_paths
            ]

            fully_personalized_params = 0
            total_valid_params = 0

            for k in valid_keys_recovery:

                recovered_sum = torch.zeros_like(
                    recovered_state[k]
                )

                recovered_count = torch.zeros_like(
                    recovered_state[k],
                    dtype=torch.float32
                )

                for state_i, client_mask, cid in zip(
                    previous_states,
                    client_masks,
                    client_ids
                ):
                    if k not in state_i:
                        print(
                            f"ERROR: Parameter '{k}' is missing from "
                            f"the previous merged model for {cid}."
                        )
                        exit(1)

                    if k not in client_mask:
                        print(
                            f"ERROR: Parameter '{k}' is missing from "
                            f"the FedSelect mask for {cid}."
                        )
                        exit(1)

                    # mask == False -> parameter is global/shared.
                    # Therefore this client's merged model contains
                    # the global parameter value here.
                    active_mask = (~client_mask[k]).float()

                    recovered_sum += (
                        state_i[k] * active_mask
                    )

                    recovered_count += active_mask

                # At least one client still has this parameter shared
                recoverable = recovered_count > 0

                fully_personalized_params += (
                    (~recoverable).sum().item()
                )

                total_valid_params += recoverable.numel()

                # Recover shared parameters exactly.
                #
                # If every client has permanently personalized a
                # parameter, its global value will never be used again
                # because FedSelect masks are cumulative. Fill with 0.
                recovered_state[k] = torch.where(
                    recoverable,
                    recovered_sum / recovered_count.clamp(min=1),
                    torch.zeros_like(recovered_state[k])
                )

            pct_fully_personalized = (
                100.0 * fully_personalized_params / total_valid_params
                if total_valid_params > 0
                else 0.0
            )

            print(
                f"Fully personalized parameters filled with zeros: "
                f"{fully_personalized_params:,}/{total_valid_params:,} "
                f"({pct_fully_personalized:.4f}%)"
            )

        # ---------------------------------------------------------
        # Save reconstructed global model
        # ---------------------------------------------------------
        os.makedirs(
            os.path.dirname(prev_global_path),
            exist_ok=True
        )

        torch.save(
            recovered_ckpt,
            prev_global_path
        )

        print(
            f"Successfully reconstructed global model from "
            f"round {previous_round} at:\n"
            f"  {prev_global_path}"
        )




    prev_global_ckpt = torch.load(prev_global_path, map_location='cpu')
    prev_state = prev_global_ckpt['state_dict']
    
    # Identify valid floating-point keys
    valid_keys = [k for k, v in prev_state.items() if v.is_floating_point() and 'num_batches_tracked' not in k]
    total_params = sum(prev_state[k].numel() for k in valid_keys)
    print(f"Total valid parameters for FedSelect: {total_params:,}")

    # ---------------------------------------------------------
    # Layer Mapping for Visualization (Created Once)
    # ---------------------------------------------------------
    mapping_file = os.path.join(mask_dir, "layer_mapping.pth")
    if not os.path.exists(mapping_file):
        layer_info = {}
        current_idx = 0
        for k in valid_keys:
            numel = prev_state[k].numel()
            layer_info[k] = {
                "shape": list(prev_state[k].shape),
                "numel": numel,
                "start_idx": current_idx,
                "end_idx": current_idx + numel
            }
            current_idx += numel
        torch.save(layer_info, mapping_file)
        print(f"Created layer mapping file at {mapping_file}")

    # ---------------------------------------------------------
    # Auto-Detect Current Round Directory
    # ---------------------------------------------------------
    # ### NEW: Scan for existing round directories and increment ###
    existing_rounds = []
    for d in os.listdir(mask_dir):
        if d.startswith("round_") and os.path.isdir(os.path.join(mask_dir, d)):
            try:
                existing_rounds.append(int(d.split("_")[1]))
            except ValueError:
                pass
    current_round = max(existing_rounds) + 1 if existing_rounds else 0
    round_mask_dir = os.path.join(mask_dir, f"round_{current_round}")
    os.makedirs(round_mask_dir, exist_ok=True)
    print(f"Saving historical masks for round {current_round} to {round_mask_dir}")
    # ##############################################################

    # ---------------------------------------------------------
    # Phase 1: Client Subnetwork Discovery (Mask Updating)
    # ---------------------------------------------------------
    print("Phase 1: Expanding personalized client subnetworks...")
    for m_path, cid in zip(models, client_ids):
        mask_path = os.path.join(mask_dir, f"{cid}_mask.pth")
        
        # Load or initialize client mask (0 = Global, 1 = Personalized)
        if os.path.exists(mask_path):
            client_mask = torch.load(mask_path)
        else:
            client_mask = {k: torch.zeros_like(prev_state[k], dtype=torch.bool) for k in valid_keys}

        ckpt_i = torch.load(m_path, map_location='cpu')
        state_i = ckpt_i['state_dict']
        
        # Calculate parameter changes and collect valid ones for thresholding
        all_diffs = []
        current_personalized = 0
        
        for k in valid_keys:
            if k in state_i:
                diff = torch.abs(state_i[k] - prev_state[k])
                # Only evaluate parameters that are currently shared (mask == 0)
                global_mask = ~client_mask[k]
                all_diffs.append(diff[global_mask].flatten())
                current_personalized += client_mask[k].sum().item()
                
        # Determine how many new parameters to select this round
        cat_diffs = torch.cat(all_diffs)
        k_to_select = int(total_params * select_ratio)
        max_allowed = int(total_params * max_sparsity)
        k_to_select = min(k_to_select, max_allowed - current_personalized)
        
        if k_to_select > 0 and len(cat_diffs) > 0:
            k_to_select = min(k_to_select, len(cat_diffs))
            
            # --- NEW LOGIC: Calculate averages ---
            avg_overall_diff = cat_diffs.mean().item()
            top_values = torch.topk(cat_diffs, k_to_select).values
            threshold = top_values[-1]
            avg_selected_diff = top_values.mean().item()
            
            print(f"  {cid}: Avg diff of all shared weights: {avg_overall_diff:.6f} | Avg diff of selected {select_ratio*100}%: {avg_selected_diff:.6f}")
            # -------------------------------------
            
            # Update the mask permanently
            new_personalized = 0
            for k in valid_keys:
                if k in state_i:
                    diff = torch.abs(state_i[k] - prev_state[k])
                    # Flip 0 to 1 if it exceeds threshold and is currently 0
                    new_ones = (~client_mask[k]) & (diff >= threshold)
                    client_mask[k][new_ones] = True
                    new_personalized += new_ones.sum().item()
                    
            print(f"  {cid}: Personalized {new_personalized:,} new params. Total sparsity: {(current_personalized + new_personalized) / total_params * 100:.2f}%")
        else:
            print(f"  {cid}: Reached max sparsity or no params to select. Total sparsity: {current_personalized / total_params * 100:.2f}%")
            
        torch.save(client_mask, mask_path)

        # ### NEW: Save historical visualization mask (0=Global, 1=Personal) ###
        vis_mask = {k: v.to(torch.int8) for k, v in client_mask.items()}
        hist_mask_path = os.path.join(round_mask_dir, f"{cid}_mask.pt")
        torch.save(vis_mask, hist_mask_path)
        # ######################################################################

        del ckpt_i
        
    # ---------------------------------------------------------
    # Phase 2: Masked Server Aggregation
    # ---------------------------------------------------------
    print("Phase 2: Aggregating shared parameters on server...")
    running_sum = {k: torch.zeros_like(v) for k, v in prev_state.items() if k in valid_keys}
    presence_weights = {k: torch.zeros_like(v) for k, v in prev_state.items() if k in valid_keys}
    
    for m_path, cid, w_i in zip(models, client_ids, norm_weights):
        mask_path = os.path.join(mask_dir, f"{cid}_mask.pth")
        client_mask = torch.load(mask_path)
        
        ckpt_i = torch.load(m_path, map_location='cpu')
        state_i = ckpt_i['state_dict']
        
        for k in valid_keys:
            if k in state_i:
                # Calculate aggregation weight per parameter: w_i * (1 - mask)
                active_mask = (~client_mask[k]).float()
                running_sum[k] += state_i[k] * active_mask * w_i
                presence_weights[k] += active_mask * w_i
                
        del ckpt_i
        
    # Finalize averaged weights (handling division by zero for fully personalized params)
    averaged_weights = {}
    for k in valid_keys:
        valid_mask = presence_weights[k] > 0
        averaged_weights[k] = torch.where(
            valid_mask,
            running_sum[k] / presence_weights[k].clamp(min=1e-9),
            prev_state[k] # Fallback to previous global weight if all clients personalized it
        )
        
    # Save the new global model for the *next* round's difference calculation
    for k in valid_keys:
        prev_global_ckpt['state_dict'][k] = averaged_weights[k]
    torch.save(prev_global_ckpt, prev_global_path)
    del prev_global_ckpt

    # ---------------------------------------------------------
    # Phase 3: Client Subnetwork Injection & Saving
    # ---------------------------------------------------------
    print("Phase 3: Injecting shared weights into client models...")
    for in_path, out_path, cid in zip(models, output_paths, client_ids):
        ckpt = torch.load(in_path, map_location='cpu')
        state = ckpt['state_dict']
        
        mask_path = os.path.join(mask_dir, f"{cid}_mask.pth")
        client_mask = torch.load(mask_path)
        
        for k in valid_keys:
            if k in state:
                # Final weight = Mask * Local + (1 - Mask) * Global
                m = client_mask[k].float()
                state[k] = (m * state[k]) + ((1.0 - m) * averaged_weights[k])
                
        reset_optimizer_state(ckpt, no_reset_optimizer)
                        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(ckpt, out_path)
        print(f"Saved personalized FedSelect model to {out_path}")


def fedselect_elastic(models, output_paths, norm_weights, client_ids, prev_global_path="/workspace/work_dirs/fedselect_states/global_model.pth", mask_dir="/workspace/work_dirs/fedselect_masks", select_ratio=0.05, max_sparsity=0.5, no_reset_optimizer=False):
    """
    FedSelect implementation with Relative Scaling & Mask Reintroduction.
    """
    print(f"Running FedSelect Aggregation (select_ratio={select_ratio}, max_sparsity={max_sparsity})...")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(os.path.dirname(prev_global_path), exist_ok=True)

    if not os.path.exists(prev_global_path):
        print(f"No previous global model found at {prev_global_path}. Exiting. Run one round of standard FedAvg first.")
        exit(1)
        
    prev_global_ckpt = torch.load(prev_global_path, map_location='cpu')
    prev_state = prev_global_ckpt['state_dict']
    
    valid_keys = [k for k, v in prev_state.items() if v.is_floating_point() and 'num_batches_tracked' not in k]
    total_params = sum(prev_state[k].numel() for k in valid_keys)
    print(f"Total valid parameters for FedSelect: {total_params:,}")

    # ---------------------------------------------------------
    # Layer Mapping for Visualization (Created Once)
    # ---------------------------------------------------------
    mapping_file = os.path.join(mask_dir, "layer_mapping.pth")
    if not os.path.exists(mapping_file):
        layer_info = {}
        current_idx = 0
        for k in valid_keys:
            numel = prev_state[k].numel()
            layer_info[k] = {
                "shape": list(prev_state[k].shape),
                "numel": numel,
                "start_idx": current_idx,
                "end_idx": current_idx + numel
            }
            current_idx += numel
        torch.save(layer_info, mapping_file)
        print(f"Created layer mapping file at {mapping_file}")
    # ---------------------------------------------------------
    # Auto-Detect Current Round Directory
    # ---------------------------------------------------------
    # ### NEW: Scan for existing round directories and increment ###
    existing_rounds = []
    for d in os.listdir(mask_dir):
        if d.startswith("round_") and os.path.isdir(os.path.join(mask_dir, d)):
            try:
                existing_rounds.append(int(d.split("_")[1]))
            except ValueError:
                pass
    current_round = max(existing_rounds) + 1 if existing_rounds else 0
    round_mask_dir = os.path.join(mask_dir, f"round_{current_round}")
    os.makedirs(round_mask_dir, exist_ok=True)
    print(f"Saving historical masks for round {current_round} to {round_mask_dir}")
    # ##############################################################

    # ---------------------------------------------------------
    # Phase 1: Client Subnetwork Discovery & Reintroduction
    # ---------------------------------------------------------
    print("Phase 1: Updating personalized client subnetworks...")
    for m_path, cid in zip(models, client_ids):
        mask_path = os.path.join(mask_dir, f"{cid}_mask.pth")
        
        if os.path.exists(mask_path):
            client_mask = torch.load(mask_path)
        else:
            client_mask = {k: torch.zeros_like(prev_state[k], dtype=torch.bool) for k in valid_keys}

        ckpt_i = torch.load(m_path, map_location='cpu')
        state_i = ckpt_i['state_dict']
        
        all_shared_rel_diffs = []
        current_personalized = 0
        
        for k in valid_keys:
            if k in state_i:
                # --- FIX 1: Relative Difference ---
                # Add 1e-8 to denominator to prevent division by zero
                rel_diff = torch.abs(state_i[k] - prev_state[k]) / (torch.abs(prev_state[k]) + 1e-8)
                
                global_mask = ~client_mask[k]
                all_shared_rel_diffs.append(rel_diff[global_mask].flatten())
                current_personalized += client_mask[k].sum().item()
                
        cat_diffs = torch.cat(all_shared_rel_diffs)
        
        # Calculate standard mathematical slices (unaffected by sparsity caps)
        base_k = int(total_params * select_ratio)
        fetch_k = min(2 * base_k, len(cat_diffs)) 
        
        if fetch_k > 0:
            top_values = torch.topk(cat_diffs, fetch_k).values
            
            # --- FIX 2: Decouple Thresholds from Sparsity Cap ---
            
            # 1. Reintroduction Threshold (Always calculated if possible)
            next_slice = top_values[base_k:fetch_k]
            reintro_threshold = next_slice.mean().item() if len(next_slice) > 0 else -1.0
            
            # 2. Personalization Threshold (Subject to max_sparsity cap)
            remaining_budget = max(0, int(total_params * max_sparsity) - current_personalized)
            actual_k_to_select = min(base_k, remaining_budget)
            
            if actual_k_to_select > 0:
                personalize_threshold = top_values[actual_k_to_select - 1].item()
            else:
                personalize_threshold = float('inf') # Cap reached, nothing new can exceed infinity
                
            avg_overall_rel_diff = cat_diffs.mean().item()
            print(f"  {cid}: Shared Rel Diff Avg: {avg_overall_rel_diff:.6f} | Reintro Threshold: {reintro_threshold:.6f}")
            if actual_k_to_select == 0:
                print(f"  {cid}: Reached max sparsity. Only evaluating reintroductions.")
            
            # Apply changes
            new_personalized = 0
            num_reintroduced = 0
            
            for k in valid_keys:
                if k in state_i:
                    rel_diff = torch.abs(state_i[k] - prev_state[k]) / (torch.abs(prev_state[k]) + 1e-8)
                    
                    new_ones = (~client_mask[k]) & (rel_diff >= personalize_threshold)
                    
                    if reintro_threshold > 0:
                        # Notice we evaluate personalized weights (client_mask[k] == True) for reintroduction
                        reintro_zeros = client_mask[k] & (rel_diff < reintro_threshold)
                    else:
                        reintro_zeros = torch.zeros_like(client_mask[k], dtype=torch.bool)
                        
                    client_mask[k][new_ones] = True
                    client_mask[k][reintro_zeros] = False
                    
                    new_personalized += new_ones.sum().item()
                    num_reintroduced += reintro_zeros.sum().item()
                    
            new_total = current_personalized + new_personalized - num_reintroduced
            print(f"  {cid}: Personalized +{new_personalized:,} | Reintroduced -{num_reintroduced:,} | Total sparsity: {new_total / total_params * 100:.2f}%")
        else:
            print(f"  {cid}: No shared params left to evaluate. Total sparsity: {current_personalized / total_params * 100:.2f}%")
            
        torch.save(client_mask, mask_path)

        # ### NEW: Save historical visualization mask (0=Global, 1=Personal) ###
        vis_mask = {k: v.to(torch.int8) for k, v in client_mask.items()}
        hist_mask_path = os.path.join(round_mask_dir, f"{cid}_mask.pt")
        torch.save(vis_mask, hist_mask_path)
        # ######################################################################


        del ckpt_i
        
    # ---------------------------------------------------------
    # Phase 2: Masked Server Aggregation
    # ---------------------------------------------------------
    print("Phase 2: Aggregating shared parameters on server...")
    running_sum = {k: torch.zeros_like(v) for k, v in prev_state.items() if k in valid_keys}
    presence_weights = {k: torch.zeros_like(v) for k, v in prev_state.items() if k in valid_keys}
    
    for m_path, cid, w_i in zip(models, client_ids, norm_weights):
        mask_path = os.path.join(mask_dir, f"{cid}_mask.pth")
        client_mask = torch.load(mask_path)
        
        ckpt_i = torch.load(m_path, map_location='cpu')
        state_i = ckpt_i['state_dict']
        
        for k in valid_keys:
            if k in state_i:
                active_mask = (~client_mask[k]).float()
                running_sum[k] += state_i[k] * active_mask * w_i
                presence_weights[k] += active_mask * w_i
                
        del ckpt_i
        
    averaged_weights = {}
    for k in valid_keys:
        valid_mask = presence_weights[k] > 0
        averaged_weights[k] = torch.where(
            valid_mask,
            running_sum[k] / presence_weights[k].clamp(min=1e-9),
            prev_state[k] 
        )
        
    for k in valid_keys:
        prev_global_ckpt['state_dict'][k] = averaged_weights[k]
    torch.save(prev_global_ckpt, prev_global_path)
    del prev_global_ckpt

    # ---------------------------------------------------------
    # Phase 3: Client Subnetwork Injection & Saving
    # ---------------------------------------------------------
    print("Phase 3: Injecting shared weights into client models...")
    for in_path, out_path, cid in zip(models, output_paths, client_ids):
        ckpt = torch.load(in_path, map_location='cpu')
        state = ckpt['state_dict']
        
        mask_path = os.path.join(mask_dir, f"{cid}_mask.pth")
        client_mask = torch.load(mask_path)
        
        for k in valid_keys:
            if k in state:
                m = client_mask[k].float()
                state[k] = (m * state[k]) + ((1.0 - m) * averaged_weights[k])

        reset_optimizer_state(ckpt, no_reset_optimizer)
                
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(ckpt, out_path)
        print(f"Saved personalized FedSelect model to {out_path}")


def fedselect_fullelastic(models, output_paths, norm_weights, client_ids, prev_global_path="/workspace/work_dirs/fedselect_states/global_model.pth", mask_dir="/workspace/work_dirs/fedselect_masks", select_ratio=0.05, no_reset_optimizer=False):
    """
    FedSelect Full Elastic implementation.
    Evaluates all parameters from scratch every round to find the top `select_ratio` 
    most different parameters. Keeps no mask history across rounds.
    """
    print(f"Running FedSelect Full-Elastic Aggregation (select_ratio={select_ratio})...")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(os.path.dirname(prev_global_path), exist_ok=True)

    if not os.path.exists(prev_global_path):
        print(f"No previous global model found at {prev_global_path}. Exiting. Run one round of standard FedAvg first.")
        exit(1)
        
    prev_global_ckpt = torch.load(prev_global_path, map_location='cpu')
    prev_state = prev_global_ckpt['state_dict']
    
    valid_keys = [k for k, v in prev_state.items() if v.is_floating_point() and 'num_batches_tracked' not in k]
    total_params = sum(prev_state[k].numel() for k in valid_keys)
    print(f"Total valid parameters for FedSelect: {total_params:,}")

    # ---------------------------------------------------------
    # Layer Mapping for Visualization (Created Once)
    # ---------------------------------------------------------
    mapping_file = os.path.join(mask_dir, "layer_mapping.pth")
    if not os.path.exists(mapping_file):
        layer_info = {}
        current_idx = 0
        for k in valid_keys:
            numel = prev_state[k].numel()
            layer_info[k] = {
                "shape": list(prev_state[k].shape),
                "numel": numel,
                "start_idx": current_idx,
                "end_idx": current_idx + numel
            }
            current_idx += numel
        torch.save(layer_info, mapping_file)
        print(f"Created layer mapping file at {mapping_file}")

    # ---------------------------------------------------------
    # Auto-Detect Current Round Directory for History
    # ---------------------------------------------------------
    existing_rounds = []
    for d in os.listdir(mask_dir):
        if d.startswith("round_") and os.path.isdir(os.path.join(mask_dir, d)):
            try:
                existing_rounds.append(int(d.split("_")[1]))
            except ValueError:
                pass
    current_round = max(existing_rounds) + 1 if existing_rounds else 0
    round_mask_dir = os.path.join(mask_dir, f"round_{current_round}")
    os.makedirs(round_mask_dir, exist_ok=True)
    print(f"Saving historical masks for round {current_round} to {round_mask_dir}")

    # ---------------------------------------------------------
    # Phase 1: Fresh Client Subnetwork Discovery 
    # ---------------------------------------------------------
    print("Phase 1: Discovering personalized client subnetworks from scratch...")
    for m_path, cid in zip(models, client_ids):
        # Fresh mask initialized to False (0)
        client_mask = {k: torch.zeros_like(prev_state[k], dtype=torch.bool) for k in valid_keys}

        ckpt_i = torch.load(m_path, map_location='cpu')
        state_i = ckpt_i['state_dict']
        
        all_rel_diffs = []
        
        # Calculate relative differences for ALL valid parameters
        for k in valid_keys:
            if k in state_i:
                rel_diff = torch.abs(state_i[k] - prev_state[k]) / (torch.abs(prev_state[k]) + 1e-8)
                all_rel_diffs.append(rel_diff.flatten())
                
        cat_diffs = torch.cat(all_rel_diffs)
        k_to_select = int(total_params * select_ratio)
        
        if k_to_select > 0:
            # Find the threshold for the top 'select_ratio' parameters
            top_values = torch.topk(cat_diffs, k_to_select).values
            personalize_threshold = top_values[-1].item()
            
            avg_overall_rel_diff = cat_diffs.mean().item()
            print(f"  {cid}: Shared Rel Diff Avg: {avg_overall_rel_diff:.6f} | Threshold: {personalize_threshold:.6f}")
            
            new_personalized = 0
            
            # Apply the mask based purely on the new threshold
            for k in valid_keys:
                if k in state_i:
                    rel_diff = torch.abs(state_i[k] - prev_state[k]) / (torch.abs(prev_state[k]) + 1e-8)
                    new_ones = rel_diff >= personalize_threshold
                    client_mask[k] = new_ones
                    new_personalized += new_ones.sum().item()
                    
            print(f"  {cid}: Personalized {new_personalized:,} params | Total sparsity: {new_personalized / total_params * 100:.2f}%")
        else:
            print(f"  {cid}: Select ratio is 0. Total sparsity: 0.00%")
            
        # Overwrite/save the main mask for Phase 2/3
        mask_path = os.path.join(mask_dir, f"{cid}_mask.pth")
        torch.save(client_mask, mask_path)

        # Save historical visualization mask (0=Global, 1=Personal)
        vis_mask = {k: v.to(torch.int8) for k, v in client_mask.items()}
        hist_mask_path = os.path.join(round_mask_dir, f"{cid}_mask.pt")
        torch.save(vis_mask, hist_mask_path)

        del ckpt_i
        
    # ---------------------------------------------------------
    # Phase 2: Masked Server Aggregation
    # ---------------------------------------------------------
    print("Phase 2: Aggregating shared parameters on server...")
    running_sum = {k: torch.zeros_like(v) for k, v in prev_state.items() if k in valid_keys}
    presence_weights = {k: torch.zeros_like(v) for k, v in prev_state.items() if k in valid_keys}
    
    for m_path, cid, w_i in zip(models, client_ids, norm_weights):
        mask_path = os.path.join(mask_dir, f"{cid}_mask.pth")
        client_mask = torch.load(mask_path)
        
        ckpt_i = torch.load(m_path, map_location='cpu')
        state_i = ckpt_i['state_dict']
        
        for k in valid_keys:
            if k in state_i:
                active_mask = (~client_mask[k]).float()
                running_sum[k] += state_i[k] * active_mask * w_i
                presence_weights[k] += active_mask * w_i
                
        del ckpt_i
        
    averaged_weights = {}
    for k in valid_keys:
        valid_mask = presence_weights[k] > 0
        averaged_weights[k] = torch.where(
            valid_mask,
            running_sum[k] / presence_weights[k].clamp(min=1e-9),
            prev_state[k] 
        )
        
    for k in valid_keys:
        prev_global_ckpt['state_dict'][k] = averaged_weights[k]
    torch.save(prev_global_ckpt, prev_global_path)
    del prev_global_ckpt

    # ---------------------------------------------------------
    # Phase 3: Client Subnetwork Injection & Saving
    # ---------------------------------------------------------
    print("Phase 3: Injecting shared weights into client models...")
    for in_path, out_path, cid in zip(models, output_paths, client_ids):
        ckpt = torch.load(in_path, map_location='cpu')
        state = ckpt['state_dict']
        
        mask_path = os.path.join(mask_dir, f"{cid}_mask.pth")
        client_mask = torch.load(mask_path)
        
        for k in valid_keys:
            if k in state:
                m = client_mask[k].float()
                # Personal weights remain (m * state), Global weights injected ((1-m) * averaged)
                state[k] = (m * state[k]) + ((1.0 - m) * averaged_weights[k])

        reset_optimizer_state(ckpt, no_reset_optimizer)
                
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(ckpt, out_path)
        print(f"Saved personalized FedSelect model to {out_path}")


def fedselect_cka(models, output_paths, norm_weights, client_ids, prev_global_path, mask_dir, select_ratio, max_sparsity, runner_path, config, data_dirs, modality, cka_samples, dynamic_sparsity, no_reset_optimizer=False):
    """
    FedSelect CKA implementation.
    Computes CKA between the previous global model and each client model.
    Selects entire layers with the lowest CKA similarity scores until the parameter count 
    reaches `select_ratio`, keeping them fully personalized. Caps at `max_sparsity`.
    """
    print(f"Running FedSelect CKA Aggregation (select_ratio={select_ratio}, max_sparsity={max_sparsity})...")
    start_time = time.time()
    print('Starting FedSelect CKA time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(os.path.dirname(prev_global_path), exist_ok=True)

    print(f"Ensured directories exist:")
    print(f"  mask_dir={mask_dir}")
    print(f"  global_dir={os.path.dirname(prev_global_path)}")

    if not config:
        raise ValueError("FedCKA requires --config.")

    if len(data_dirs) != len(models) or any(not path for path in data_dirs):
        raise ValueError("FedCKA requires one valid data directory per client.")

    if not os.path.exists(prev_global_path):
        print(f"No previous global model found at {prev_global_path}. Exiting. Run one round of standard FedAvg first to initialize the global model.")
        exit(1)
        
    prev_global_ckpt = torch.load(prev_global_path, map_location='cpu')
    prev_state = prev_global_ckpt['state_dict']
    
    valid_keys = [k for k, v in prev_state.items() if v.is_floating_point() and 'num_batches_tracked' not in k]
    print(f"Valid floating-point keys used for masking/aggregation: {len(valid_keys)}")
    for k in valid_keys[:10]:
        print(f"  {k} | shape={tuple(prev_state[k].shape)} | numel={prev_state[k].numel()}")

    total_params = sum(prev_state[k].numel() for k in valid_keys)
    print(f"Total valid parameters: {total_params:,}")

    print(f"Max allowed personalized params per client: {int(total_params * max_sparsity):,}")
    print(f"Target params to newly personalize per round: {int(total_params * select_ratio):,}")

    # 1. Dynamically import runner.py
    if not os.path.exists(runner_path):
        raise FileNotFoundError(f"CKA runner script not found at {runner_path}")
    
    spec = importlib.util.spec_from_file_location("cka_runner", runner_path)
    cka_runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cka_runner)

    # ---------------------------------------------------------
    # Auto-Detect Current Round Directory for History
    # ---------------------------------------------------------
    existing_rounds = []
    for d in os.listdir(mask_dir):
        if d.startswith("round_") and os.path.isdir(os.path.join(mask_dir, d)):
            try:
                existing_rounds.append(int(d.split("_")[1]))
            except ValueError:
                pass
    current_round = max(existing_rounds) + 1 if existing_rounds else 0
    print(f"Detected previous rounds: {sorted(existing_rounds)}")
    print(f"Current round inferred as: {current_round}")

    round_mask_dir = os.path.join(mask_dir, f"round_{current_round}")
    os.makedirs(round_mask_dir, exist_ok=True)
    print(f"Saving historical masks for round {current_round} to {round_mask_dir}")

    # 2. Phase 1: Client Subnetwork Discovery via CKA
    print("\nPhase 1: Discovering personalized client layers via CKA...")
    start_time_cka = time.time()
    print('Starting CKA time: ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time_cka)))
    for m_path, cid, data_dir in zip(models, client_ids, data_dirs):
        print(f"\n==================== CLIENT {cid} ====================")
        print(f"Client checkpoint path: {m_path}")

        mask_path = os.path.join(mask_dir, f"{cid}_mask.pth")
        
        # Load existing mask to preserve history across rounds
        if os.path.exists(mask_path):
            client_mask = torch.load(mask_path)
        else:
            client_mask = {k: torch.zeros_like(prev_state[k], dtype=torch.bool) for k in valid_keys}

        current_personalized = sum(client_mask[k].sum().item() for k in valid_keys)
        
        if dynamic_sparsity == 0.0:
            print(f"{cid}: Dynamic sparsity disabled. Masking layers sequentially based on CKA until budget is met.")
        
            # Calculate parameter budget for this round
            max_allowed = int(total_params * max_sparsity)
            k_to_select = int(total_params * select_ratio)
            k_to_select = min(k_to_select, max_allowed - current_personalized)
        
            print(f"{cid}: current_personalized={current_personalized:,} / {total_params:,} ({current_personalized / total_params * 100:.2f}%)")
            print(f"{cid}: max_allowed={max_allowed:,}")
            print(f"{cid}: remaining personalization budget={max_allowed - current_personalized:,}")
            print(f"{cid}: final k_to_select={k_to_select:,}")

            if k_to_select > 0:
                print(f"\n--- Computing CKA for {cid} ---")
                cka_rows = cka_runner.run_cka(
                    config=config,
                    data_dir=data_dir,
                    checkpoints=[prev_global_path, m_path],
                    modality=modality,
                    max_samples=cka_samples,
                    output_dir=os.path.join(mask_dir, "cka_temp")
                )

                # Filter out NaNs and sort by CKA similarity ASCENDING (Lowest similarity first)
                valid_cka = [(idx, name, score) for idx, name, score in cka_rows if not math.isnan(score)]
                print(f"{cid}: CKA returned {len(cka_rows)} rows.")
                print(f"{cid}: valid CKA rows={len(valid_cka)}, NaN rows filtered out={len(cka_rows) - len(valid_cka)}")

                valid_cka.sort(key=lambda x: x[2])
                print(f"{cid}: 10 lowest-similarity layers by CKA:")
                for idx, name, score in valid_cka[:10]:
                    print(f"  idx={idx} | layer={name} | cka={score:.6f}")
                
                print(f"{cid}: 10 highest-similarity layers by CKA:")
                for idx, name, score in valid_cka[-10:]:
                    print(f"  idx={idx} | layer={name} | cka={score:.6f}")
                    
                new_personalized = 0
                layers_masked = 0
            
                # Iterate through layers with lowest CKA similarity scores
                for idx, layer_name, score in valid_cka:
                    # Find all parameter keys belonging to this layer
                    layer_keys = [k for k in valid_keys if k.replace('module.', '').startswith(layer_name + '.') or k.replace('module.', '') == layer_name]
                    
                    # Count parameters in this layer that are NOT YET personalized
                    layer_new_params = sum((~client_mask[k]).sum().item() for k in layer_keys)
                    
                    if layer_new_params > 0:
                        print(f"{cid}: SELECTING layer {layer_name} with CKA={score:.6f}, adding {layer_new_params:,} new params")

                        # Apply mask to the entire layer
                        for k in layer_keys:
                            # Convert 0s to 1s
                            client_mask[k] = torch.ones_like(client_mask[k], dtype=torch.bool)
                            
                        new_personalized += layer_new_params
                        layers_masked += 1
                        
                        # Stop if we have met or slightly exceeded our parameter budget for this round
                        if new_personalized >= k_to_select:
                            break
                            
                print(f"  {cid}: Masked {layers_masked} entire layers.")
                print(f"  {cid}: Personalized {new_personalized:,} new params. Total sparsity: {(current_personalized + new_personalized) / total_params * 100:.2f}%")
            else:
                print(f"  {cid}: Reached max sparsity ({max_sparsity*100}%). Skipping CKA computation. Total sparsity: {current_personalized / total_params * 100:.2f}%")

        # dynamic sparsity mode: mask all parameters with CKA scores above the threshold, ignoring the select_ratio budget
        else:
            print(f"\n--- Computing CKA for {cid} ---")
            cka_rows = cka_runner.run_cka(
                config=config,
                data_dir=data_dir,
                checkpoints=[prev_global_path, m_path],
                modality=modality,
                max_samples=cka_samples,
                output_dir=os.path.join(mask_dir, "cka_temp")
            )

            # Filter out NaNs and sort by CKA similarity ASCENDING (Lowest similarity first)
            valid_cka = [(idx, name, score) for idx, name, score in cka_rows if not math.isnan(score)]
            print(f"{cid}: CKA returned {len(cka_rows)} rows.")
            print(f"{cid}: valid CKA rows={len(valid_cka)}, NaN rows filtered out={len(cka_rows) - len(valid_cka)}")

            valid_cka.sort(key=lambda x: x[2])
            print(f"{cid}: 10 lowest-similarity layers by CKA:")
            for idx, name, score in valid_cka[:10]:
                print(f"  idx={idx} | layer={name} | cka={score:.6f}")
            
            print(f"{cid}: 10 highest-similarity layers by CKA:")
            for idx, name, score in valid_cka[-10:]:
                print(f"  idx={idx} | layer={name} | cka={score:.6f}")

            new_personalized = 0
            layers_masked = 0
        
            # Iterate through layers with lowest CKA similarity scores
            for idx, layer_name, score in valid_cka:
                
                # Check against the dynamic threshold
                if score < dynamic_sparsity:
                    # Find all parameter keys belonging to this layer
                    layer_keys = [k for k in valid_keys if k.replace('module.', '').startswith(layer_name + '.') or k.replace('module.', '') == layer_name]
                    
                    # Count parameters in this layer that are NOT YET personalized
                    layer_new_params = sum((~client_mask[k]).sum().item() for k in layer_keys)
                    
                    if layer_new_params > 0:
                        print(f"{cid}: SELECTING layer {layer_name} with CKA similarity={score:.6f} < {dynamic_sparsity}, adding {layer_new_params:,} new params")

                        # Apply mask to the entire layer
                        for k in layer_keys:
                            # Convert 0s to 1s
                            client_mask[k] = torch.ones_like(client_mask[k], dtype=torch.bool)
                            
                        new_personalized += layer_new_params
                        layers_masked += 1
                else:
                    # Since valid_cka is sorted ascending, all subsequent layers have higher similarity.
                    print(f"{cid}: CKA similarity {score:.6f} reached threshold {dynamic_sparsity}. Stopping selection.")
                    break
                        
            print(f"  {cid}: Masked {layers_masked} entire layers based on threshold.")
            print(f"  {cid}: Personalized {new_personalized:,} new params. Total sparsity: {(current_personalized + new_personalized) / total_params * 100:.2f}%")


        torch.save(client_mask, mask_path)
        
        # Save historical visualization mask (0=Global, 1=Personal)
        vis_mask = {k: v.to(torch.int8) for k, v in client_mask.items()}
        hist_mask_path = os.path.join(round_mask_dir, f"{cid}_mask.pt")
        torch.save(vis_mask, hist_mask_path)
    end_time_cka = time.time()
    print('CKA completed in {:.2f} seconds'.format(end_time_cka - start_time_cka))
    print('FedSelect CKA Phase 1 completed in {:.2f} seconds'.format(end_time_cka - start_time))


    # 3. Phase 2: Masked Server Aggregation
    print("\nPhase 2: Aggregating shared parameters on server...")
    running_sum = {k: torch.zeros_like(v) for k, v in prev_state.items() if k in valid_keys}
    presence_weights = {k: torch.zeros_like(v) for k, v in prev_state.items() if k in valid_keys}
    
    for m_path, cid, w_i in zip(models, client_ids, norm_weights):
        mask_path = os.path.join(mask_dir, f"{cid}_mask.pth")
        client_mask = torch.load(mask_path)
        
        ckpt_i = torch.load(m_path, map_location='cpu')
        state_i = ckpt_i['state_dict']
        
        for k in valid_keys:
            if k in state_i:
                active_mask = (~client_mask[k]).float()
                running_sum[k] += state_i[k] * active_mask * w_i
                presence_weights[k] += active_mask * w_i
                
        del ckpt_i
        
    averaged_weights = {}
    for k in valid_keys:
        valid_mask = presence_weights[k] > 0
        averaged_weights[k] = torch.where(
            valid_mask,
            running_sum[k] / presence_weights[k].clamp(min=1e-9),
            prev_state[k] # Fallback to previous global if all clients personalized it
        )
    
    num_fallback_keys = sum((presence_weights[k] <= 0).all().item() for k in valid_keys)
    print(f"Keys fully falling back to previous global because all clients personalized them: {num_fallback_keys}")

    for k in valid_keys:
        prev_global_ckpt['state_dict'][k] = averaged_weights[k]
    torch.save(prev_global_ckpt, prev_global_path)
    del prev_global_ckpt

    end_time_agg = time.time()
    print('FedSelect CKA Phase 2 completed in {:.2f} seconds'.format(end_time_agg - end_time_cka))
    print('FedSelect CKA Phase 1+2 completed in {:.2f} seconds'.format(end_time_agg - start_time))

    # 4. Phase 3: Client Subnetwork Injection & Saving
    print("Phase 3: Injecting shared weights into client models...")
    for in_path, out_path, cid in zip(models, output_paths, client_ids):
        ckpt = torch.load(in_path, map_location='cpu')
        state = ckpt['state_dict']
        
        mask_path = os.path.join(mask_dir, f"{cid}_mask.pth")
        client_mask = torch.load(mask_path)
        
        for k in valid_keys:
            if k in state:
                m = client_mask[k].float()
                state[k] = (m * state[k]) + ((1.0 - m) * averaged_weights[k])

        reset_optimizer_state(ckpt, no_reset_optimizer)
                
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(ckpt, out_path)
        print(f"Saved personalized FedSelect CKA model to {out_path}")

    end_time = time.time()
    print('FedSelect CKA completed in {:.2f} seconds'.format(end_time - start_time))
    end_time_subnet = time.time()
    print('FedSelect CKA Phase 3 completed in {:.2f} seconds'.format(end_time_subnet - end_time_agg))
    print('FedSelect CKA Phase 1+2+3 completed in {:.2f} seconds'.format(end_time_subnet - start_time))
    
    # ---------------------------------------------------------
    # Phase 4: Cleanup .pth files older than 2 rounds ago
    # ---------------------------------------------------------
    print("\nPhase 4: Cleaning up old .pth checkpoint files...")
    
    # FIX: Step up one extra level to reach the root workspace ($WORK) instead of fedselect_states/
    base_search_dir = os.path.dirname(os.path.dirname(prev_global_path))
    
    # FIX: Independently verify the true highest round directly from the workspace folders
    # This prevents the cleanup from failing if the script's earlier 'current_round' variable is desynced
    existing_actual_rounds = []
    if os.path.exists(base_search_dir):
        for d in os.listdir(base_search_dir):
            if d.startswith("round_") and os.path.isdir(os.path.join(base_search_dir, d)):
                try:
                    existing_actual_rounds.append(int(d.split("_")[1]))
                except ValueError:
                    pass
                    
    actual_current_round = max(existing_actual_rounds) if existing_actual_rounds else current_round
    threshold_round = actual_current_round - 2
    
    print(f"Base search dir for cleanup: {base_search_dir}")
    print(f"Highest detected round: {actual_current_round} | Deleting files older than round: {threshold_round}")

    if threshold_round >= 0:
        deleted_count = 0
        for root, dirs, files in os.walk(base_search_dir):
            # Check if we are currently inside any folder named 'round_X'
            round_folder = next((p for p in root.split(os.sep) if p.startswith("round_")), None)
            
            if round_folder:
                try:
                    r_num = int(round_folder.split("_")[1])
                    # If the round is older than actual_current_round - 2, clear its .pth files ONLY
                    if r_num < threshold_round:
                        for f in files:
                            if f.endswith(".pth"):
                                target_file = os.path.join(root, f)
                                os.remove(target_file)
                                deleted_count += 1
                                print(f"  🗑️ Deleted old round file: {target_file}")
                except ValueError:
                    pass # Ignore folders that are named "round_something" but aren't numbers
                    
        print(f"Cleanup complete. Removed {deleted_count} old .pth files.")
    else:
        print(f"Current round is {actual_current_round}. Threshold is {threshold_round}. No cleanup needed yet.")
    
def fedselect_cka_elastic(models, output_paths, norm_weights, client_ids, prev_global_path, mask_dir, select_ratio, max_sparsity, runner_path, config, data_dirs, modality, cka_samples, dynamic_sparsity, no_reset_optimizer=False):
    """
    FedSelect CKA implementation.
    Computes CKA between the previous global model and each client model.
    Selects entire layers with the lowest CKA similarity scores until the parameter count 
    reaches `select_ratio`, keeping them fully personalized. Caps at `max_sparsity`.
    """
    print(f"Running FedSelect CKA Aggregation (select_ratio={select_ratio}, max_sparsity={max_sparsity})...")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(os.path.dirname(prev_global_path), exist_ok=True)

    print(f"Ensured directories exist:")
    print(f"  mask_dir={mask_dir}")
    print(f"  global_dir={os.path.dirname(prev_global_path)}")

    if not config:
        raise ValueError("FedCKA requires --config.")

    if len(data_dirs) != len(models) or any(not path for path in data_dirs):
        raise ValueError("FedCKA requires one valid data directory per client.")

    if not os.path.exists(prev_global_path):
        print(f"No previous global model found at {prev_global_path}. Exiting. Run one round of standard FedAvg first to initialize the global model.")
        exit(1)
        
    prev_global_ckpt = torch.load(prev_global_path, map_location='cpu')
    prev_state = prev_global_ckpt['state_dict']
    
    valid_keys = [k for k, v in prev_state.items() if v.is_floating_point() and 'num_batches_tracked' not in k]
    print(f"Valid floating-point keys used for masking/aggregation: {len(valid_keys)}")
    for k in valid_keys[:10]:
        print(f"  {k} | shape={tuple(prev_state[k].shape)} | numel={prev_state[k].numel()}")

    total_params = sum(prev_state[k].numel() for k in valid_keys)
    print(f"Total valid parameters: {total_params:,}")

    print(f"Max allowed personalized params per client: {int(total_params * max_sparsity):,}")
    print(f"Target params to newly personalize per round: {int(total_params * select_ratio):,}")

    # 1. Dynamically import runner.py
    if not os.path.exists(runner_path):
        raise FileNotFoundError(f"CKA runner script not found at {runner_path}")
    
    spec = importlib.util.spec_from_file_location("cka_runner", runner_path)
    cka_runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cka_runner)

    # ---------------------------------------------------------
    # Auto-Detect Current Round Directory for History
    # ---------------------------------------------------------
    existing_rounds = []
    for d in os.listdir(mask_dir):
        if d.startswith("round_") and os.path.isdir(os.path.join(mask_dir, d)):
            try:
                existing_rounds.append(int(d.split("_")[1]))
            except ValueError:
                pass
    current_round = max(existing_rounds) + 1 if existing_rounds else 0
    print(f"Detected previous rounds: {sorted(existing_rounds)}")
    print(f"Current round inferred as: {current_round}")

    round_mask_dir = os.path.join(mask_dir, f"round_{current_round}")
    os.makedirs(round_mask_dir, exist_ok=True)
    print(f"Saving historical masks for round {current_round} to {round_mask_dir}")

    # 2. Phase 1: Client Subnetwork Discovery via CKA
    print("\nPhase 1: Discovering personalized client layers via CKA...")
    for m_path, cid, data_dir in zip(models, client_ids, data_dirs):
        print(f"\n==================== CLIENT {cid} ====================")
        print(f"Client checkpoint path: {m_path}")

        mask_path = os.path.join(mask_dir, f"{cid}_mask.pth")
        valid_cka = []

        # Load existing mask to preserve history across rounds
        if os.path.exists(mask_path):
            client_mask = torch.load(mask_path)
        else:
            client_mask = {k: torch.zeros_like(prev_state[k], dtype=torch.bool) for k in valid_keys}

        # Snapshot mask before this round so elastic reversions can be detected afterwards
        mask_before_round = {k: client_mask[k].clone() for k in valid_keys}

        current_personalized = sum(client_mask[k].sum().item() for k in valid_keys)
        previous_personalized = current_personalized
        
        if dynamic_sparsity == 0.0:
            print(f"{cid}: Dynamic sparsity disabled. Masking layers sequentially based on CKA until budget is met.")
        
            # Calculate parameter budget for this round
            max_allowed = int(total_params * max_sparsity)
            k_to_select = int(total_params * select_ratio)
            k_to_select = min(k_to_select, max_allowed - current_personalized)
        
            print(f"{cid}: current_personalized={current_personalized:,} / {total_params:,} ({current_personalized / total_params * 100:.2f}%)")
            print(f"{cid}: max_allowed={max_allowed:,}")
            print(f"{cid}: remaining personalization budget={max_allowed - current_personalized:,}")
            print(f"{cid}: final k_to_select={k_to_select:,}")

            if k_to_select > 0 or current_personalized > 0:
                print(f"\n--- Computing CKA for {cid} ---")
                cka_rows = cka_runner.run_cka(
                    config=config,
                    data_dir=data_dir,
                    checkpoints=[prev_global_path, m_path],
                    modality=modality,
                    max_samples=cka_samples,
                    output_dir=os.path.join(mask_dir, "cka_temp")
                )

                # Filter out NaNs and sort by CKA similarity ASCENDING (Lowest similarity first)
                valid_cka = [(idx, name, score) for idx, name, score in cka_rows if not math.isnan(score)]
                print(f"{cid}: CKA returned {len(cka_rows)} rows.")
                print(f"{cid}: valid CKA rows={len(valid_cka)}, NaN rows filtered out={len(cka_rows) - len(valid_cka)}")

                valid_cka.sort(key=lambda x: x[2])
                print(f"{cid}: 10 lowest-similarity layers by CKA:")
                for idx, name, score in valid_cka[:10]:
                    print(f"  idx={idx} | layer={name} | cka={score:.6f}")
                
                print(f"{cid}: 10 highest-similarity layers by CKA:")
                for idx, name, score in valid_cka[-10:]:
                    print(f"  idx={idx} | layer={name} | cka={score:.6f}")
                    
                previous_mask = {k: client_mask[k].clone() for k in valid_keys}

                elastic_scored_keys = set()
                elastic_layer_keys = {}
                for _, layer_name, _ in valid_cka:
                    layer_keys = [k for k in valid_keys if k.replace('module.', '').startswith(layer_name + '.') or k.replace('module.', '') == layer_name]
                    elastic_layer_keys[layer_name] = layer_keys
                    elastic_scored_keys.update(layer_keys)

                elastic_fixed_personalized = sum(
                    previous_mask[k].sum().item()
                    for k in valid_keys
                    if k not in elastic_scored_keys
                )
                elastic_target_total = min(
                    max_allowed,
                    previous_personalized + max(k_to_select, 0)
                )
                elastic_target_scored = max(
                    0,
                    elastic_target_total - elastic_fixed_personalized
                )

                elastic_selected_layers = set()
                elastic_selected_keys = set()
                elastic_selected_param_count = 0

                if elastic_target_scored > 0:
                    for _, layer_name, _ in valid_cka:
                        layer_keys = elastic_layer_keys[layer_name]
                        layer_unseen_keys = [k for k in layer_keys if k not in elastic_selected_keys]

                        if layer_unseen_keys:
                            elastic_selected_layers.add(layer_name)
                            elastic_selected_keys.update(layer_unseen_keys)
                            elastic_selected_param_count += sum(prev_state[k].numel() for k in layer_unseen_keys)

                        if elastic_selected_param_count >= elastic_target_scored:
                            break

                for k in elastic_scored_keys:
                    client_mask[k] = torch.zeros_like(client_mask[k], dtype=torch.bool)

                for k in elastic_selected_keys:
                    client_mask[k] = previous_mask[k].clone()

                current_personalized = sum(client_mask[k].sum().item() for k in valid_keys)

                new_personalized = 0
                layers_masked = 0
            
                # Iterate through layers with lowest CKA similarity scores
                for idx, layer_name, score in valid_cka:
                    # Find all parameter keys belonging to this layer
                    layer_keys = [k for k in valid_keys if k.replace('module.', '').startswith(layer_name + '.') or k.replace('module.', '') == layer_name]
                    
                    if layer_name not in elastic_selected_layers:
                        continue

                    # Count parameters in this layer that are NOT YET personalized
                    layer_new_params = sum((~client_mask[k]).sum().item() for k in layer_keys)
                    
                    if layer_new_params > 0:
                        print(f"{cid}: SELECTING layer {layer_name} with CKA={score:.6f}, adding {layer_new_params:,} new params")

                        # Apply mask to the entire layer
                        for k in layer_keys:
                            # Convert 0s to 1s
                            client_mask[k] = torch.ones_like(client_mask[k], dtype=torch.bool)
                            
                        new_personalized += layer_new_params
                        layers_masked += 1
                        
                        # Stop if we have met or slightly exceeded our parameter budget for this round
                        if (current_personalized + new_personalized) >= elastic_target_total:
                            break
                            
                print(f"  {cid}: Masked {layers_masked} entire layers.")
                print(f"  {cid}: Personalized {new_personalized:,} new params. Total sparsity: {(current_personalized + new_personalized) / total_params * 100:.2f}%")
            else:
                print(f"  {cid}: Reached max sparsity ({max_sparsity*100}%). Skipping CKA computation. Total sparsity: {current_personalized / total_params * 100:.2f}%")

        # dynamic sparsity mode: mask all parameters with CKA scores above the threshold, ignoring the select_ratio budget
        else:
            print(f"\n--- Computing CKA for {cid} ---")
            cka_rows = cka_runner.run_cka(
                config=config,
                data_dir=data_dir,
                checkpoints=[prev_global_path, m_path],
                modality=modality,
                max_samples=cka_samples,
                output_dir=os.path.join(mask_dir, "cka_temp")
            )

            # Filter out NaNs and sort by CKA similarity ASCENDING (Lowest similarity first)
            valid_cka = [(idx, name, score) for idx, name, score in cka_rows if not math.isnan(score)]
            print(f"{cid}: CKA returned {len(cka_rows)} rows.")
            print(f"{cid}: valid CKA rows={len(valid_cka)}, NaN rows filtered out={len(cka_rows) - len(valid_cka)}")

            valid_cka.sort(key=lambda x: x[2])
            print(f"{cid}: 10 lowest-similarity layers by CKA:")
            for idx, name, score in valid_cka[:10]:
                print(f"  idx={idx} | layer={name} | cka={score:.6f}")
            
            print(f"{cid}: 10 highest-similarity layers by CKA:")
            for idx, name, score in valid_cka[-10:]:
                print(f"  idx={idx} | layer={name} | cka={score:.6f}")

            for _, layer_name, _ in valid_cka:
                layer_keys = [k for k in valid_keys if k.replace('module.', '').startswith(layer_name + '.') or k.replace('module.', '') == layer_name]
                for k in layer_keys:
                    client_mask[k] = torch.zeros_like(client_mask[k], dtype=torch.bool)

            current_personalized = sum(client_mask[k].sum().item() for k in valid_keys)

            new_personalized = 0
            layers_masked = 0
        
            # Iterate through layers with lowest CKA similarity scores
            for idx, layer_name, score in valid_cka:
                
                # Check against the dynamic threshold
                if score < dynamic_sparsity:
                    # Find all parameter keys belonging to this layer
                    layer_keys = [k for k in valid_keys if k.replace('module.', '').startswith(layer_name + '.') or k.replace('module.', '') == layer_name]
                    
                    # Count parameters in this layer that are NOT YET personalized
                    layer_new_params = sum((~client_mask[k]).sum().item() for k in layer_keys)
                    
                    if layer_new_params > 0:
                        print(f"{cid}: SELECTING layer {layer_name} with CKA similarity={score:.6f} < {dynamic_sparsity}, adding {layer_new_params:,} new params")

                        # Apply mask to the entire layer
                        for k in layer_keys:
                            # Convert 0s to 1s
                            client_mask[k] = torch.ones_like(client_mask[k], dtype=torch.bool)
                            
                        new_personalized += layer_new_params
                        layers_masked += 1
                else:
                    # Since valid_cka is sorted ascending, all subsequent layers have higher similarity.
                    print(f"{cid}: CKA similarity {score:.6f} reached threshold {dynamic_sparsity}. Stopping selection.")
                    break
                        
            print(f"  {cid}: Masked {layers_masked} entire layers based on threshold.")
            print(f"  {cid}: Personalized {new_personalized:,} new params. Total sparsity: {(current_personalized + new_personalized) / total_params * 100:.2f}%")


        # ---------------------------------------------------------
        # Elastic Reversion Logging
        # ---------------------------------------------------------
        reversion_log_path = os.path.join(mask_dir, "elastic_reversion_log.txt")
        cka_layer_log_path = os.path.join(mask_dir, "cka_layer_log.txt")
        mask_history_log_path = os.path.join(mask_dir, "cka_mask_history.txt")

        newly_personalized_layers = []
        newly_reglobalized_layers = []
        current_personalized_layers = []
        layer_log_lines = []

        # Use the CKA layer names from this pass, because personalization/reversion
        # is performed at exactly this layer granularity.
        if valid_cka:
            seen_layers = set()

            for _, layer_name, score in valid_cka:
                if layer_name in seen_layers:
                    continue
                seen_layers.add(layer_name)

                layer_keys = [
                    k for k in valid_keys
                    if k.replace('module.', '').startswith(layer_name + '.')
                    or k.replace('module.', '') == layer_name
                ]

                if not layer_keys:
                    continue

                # Count personalized parameters before and after this pass.
                # This is slightly more robust than just using any(), because it
                # also exposes an unexpected partial layer mask if one ever occurs.
                layer_total_params = sum(mask_before_round[k].numel() for k in layer_keys)

                personalized_before = sum(
                    mask_before_round[k].sum().item()
                    for k in layer_keys
                )

                personalized_after = sum(
                    client_mask[k].sum().item()
                    for k in layer_keys
                )

                was_personalized = personalized_before > 0
                is_personalized = personalized_after > 0

                # Determine readable current mask status.
                if personalized_after == 0:
                    mask_status = "GLOBAL"
                elif personalized_after == layer_total_params:
                    mask_status = "PERSONALIZED"
                else:
                    mask_status = (
                        f"PARTIAL({personalized_after}/{layer_total_params})"
                    )

                if is_personalized:
                    current_personalized_layers.append(layer_name)

                # Global -> Personal
                if not was_personalized and is_personalized:
                    newly_personalized_layers.append(layer_name)

                # Personal -> Global
                if was_personalized and not is_personalized:
                    newly_reglobalized_layers.append(layer_name)

                # Log CKA score + resulting current mask for EVERY CKA layer.
                layer_log_line = (
                    f"round={current_round} | "
                    f"model={cid} | "
                    f"layer={layer_name} | "
                    f"cka={score:.6f} | "
                    f"mask={mask_status}"
                )

                layer_log_lines.append(layer_log_line)

        # ---------------------------------------------------------
        # Print + save per-layer CKA and mask information
        # ---------------------------------------------------------
        for line in layer_log_lines:
            print(line)

        if layer_log_lines:
            try:
                with open(cka_layer_log_path, "a", encoding="utf-8") as f:
                    for line in layer_log_lines:
                        f.write(line + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            except OSError as e:
                print(f"WARNING: Could not write CKA layer log: {e}")

        # ---------------------------------------------------------
        # Reversion statistics
        # ---------------------------------------------------------
        reverted_layers = newly_reglobalized_layers
        reverted_this_round = len(reverted_layers)

        # Recover cumulative number of previous reversions for this client
        # from the existing text log. This survives process restarts.
        previous_total_reverted = 0

        if os.path.exists(reversion_log_path):
            try:
                with open(reversion_log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if f"model={cid} |" not in line:
                            continue

                        try:
                            logged_round = int(
                                line.split("round=")[1].split("|")[0].strip()
                            )

                            logged_total = int(
                                line.split("total_reverted=")[1]
                                .split("|")[0]
                                .strip()
                            )

                            # Only use genuinely previous rounds.
                            # This prevents accidentally double-counting if
                            # the same client/round is logged twice after a restart.
                            if logged_round < current_round:
                                previous_total_reverted = max(
                                    previous_total_reverted,
                                    logged_total
                                )

                        except (IndexError, ValueError):
                            # Ignore malformed/incomplete historical lines
                            continue

            except OSError as e:
                print(f"WARNING: Could not read elastic reversion log: {e}")

        total_reverted = previous_total_reverted + reverted_this_round

        reverted_names = (
            ", ".join(reverted_layers)
            if reverted_layers else "None"
        )

        newly_personalized_names = (
            ", ".join(newly_personalized_layers)
            if newly_personalized_layers else "None"
        )

        current_personalized_names = (
            ", ".join(current_personalized_layers)
            if current_personalized_layers else "None"
        )

        # ---------------------------------------------------------
        # Reversion summary
        # ---------------------------------------------------------
        reversion_log_line = (
            f"round={current_round} | "
            f"model={cid} | "
            f"reverted_this_round={reverted_this_round} | "
            f"total_reverted={total_reverted} | "
            f"reverted_layers=[{reverted_names}]"
        )

        print(reversion_log_line)

        try:
            with open(reversion_log_path, "a", encoding="utf-8") as f:
                f.write(reversion_log_line + "\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError as e:
            print(f"WARNING: Could not write elastic reversion log: {e}")

        # ---------------------------------------------------------
        # Complete mask-change summary
        # ---------------------------------------------------------
        mask_history_line = (
            f"round={current_round} | "
            f"model={cid} | "
            f"current_personalized=[{current_personalized_names}] | "
            f"newly_personalized=[{newly_personalized_names}] | "
            f"newly_reglobalized=[{reverted_names}]"
        )

        print(mask_history_line)

        try:
            with open(mask_history_log_path, "a", encoding="utf-8") as f:
                f.write(mask_history_line + "\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError as e:
            print(f"WARNING: Could not write CKA mask history log: {e}")

        torch.save(client_mask, mask_path)
        
        # Save historical visualization mask (0=Global, 1=Personal)
        vis_mask = {k: v.to(torch.int8) for k, v in client_mask.items()}
        hist_mask_path = os.path.join(round_mask_dir, f"{cid}_mask.pt")
        torch.save(vis_mask, hist_mask_path)

    # 3. Phase 2: Masked Server Aggregation
    print("\nPhase 2: Aggregating shared parameters on server...")
    running_sum = {k: torch.zeros_like(v) for k, v in prev_state.items() if k in valid_keys}
    presence_weights = {k: torch.zeros_like(v) for k, v in prev_state.items() if k in valid_keys}
    
    for m_path, cid, w_i in zip(models, client_ids, norm_weights):
        mask_path = os.path.join(mask_dir, f"{cid}_mask.pth")
        client_mask = torch.load(mask_path)
        
        ckpt_i = torch.load(m_path, map_location='cpu')
        state_i = ckpt_i['state_dict']
        
        for k in valid_keys:
            if k in state_i:
                active_mask = (~client_mask[k]).float()
                running_sum[k] += state_i[k] * active_mask * w_i
                presence_weights[k] += active_mask * w_i
                
        del ckpt_i
        
    averaged_weights = {}
    for k in valid_keys:
        valid_mask = presence_weights[k] > 0
        averaged_weights[k] = torch.where(
            valid_mask,
            running_sum[k] / presence_weights[k].clamp(min=1e-9),
            prev_state[k] # Fallback to previous global if all clients personalized it
        )
    
    num_fallback_keys = sum((presence_weights[k] <= 0).all().item() for k in valid_keys)
    print(f"Keys fully falling back to previous global because all clients personalized them: {num_fallback_keys}")

    for k in valid_keys:
        prev_global_ckpt['state_dict'][k] = averaged_weights[k]
    torch.save(prev_global_ckpt, prev_global_path)
    del prev_global_ckpt

    # 4. Phase 3: Client Subnetwork Injection & Saving
    print("Phase 3: Injecting shared weights into client models...")
    for in_path, out_path, cid in zip(models, output_paths, client_ids):
        ckpt = torch.load(in_path, map_location='cpu')
        state = ckpt['state_dict']
        
        mask_path = os.path.join(mask_dir, f"{cid}_mask.pth")
        client_mask = torch.load(mask_path)
        
        for k in valid_keys:
            if k in state:
                m = client_mask[k].float()
                state[k] = (m * state[k]) + ((1.0 - m) * averaged_weights[k])

        reset_optimizer_state(ckpt, no_reset_optimizer)
                
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(ckpt, out_path)
        print(f"Saved personalized FedSelect CKA model to {out_path}")
    
    # ---------------------------------------------------------
    # Phase 4: Cleanup .pth files older than 2 rounds ago
    # ---------------------------------------------------------
    print("\nPhase 4: Cleaning up old .pth checkpoint files...")
    
    # FIX: Step up one extra level to reach the root workspace ($WORK) instead of fedselect_states/
    base_search_dir = os.path.dirname(os.path.dirname(prev_global_path))
    
    # FIX: Independently verify the true highest round directly from the workspace folders
    # This prevents the cleanup from failing if the script's earlier 'current_round' variable is desynced
    existing_actual_rounds = []
    if os.path.exists(base_search_dir):
        for d in os.listdir(base_search_dir):
            if d.startswith("round_") and os.path.isdir(os.path.join(base_search_dir, d)):
                try:
                    existing_actual_rounds.append(int(d.split("_")[1]))
                except ValueError:
                    pass
                    
    actual_current_round = max(existing_actual_rounds) if existing_actual_rounds else current_round
    threshold_round = actual_current_round - 2
    
    print(f"Base search dir for cleanup: {base_search_dir}")
    print(f"Highest detected round: {actual_current_round} | Deleting files older than round: {threshold_round}")

    if threshold_round >= 0:
        deleted_count = 0
        for root, dirs, files in os.walk(base_search_dir):
            # Check if we are currently inside any folder named 'round_X'
            round_folder = next((p for p in root.split(os.sep) if p.startswith("round_")), None)
            
            if round_folder:
                try:
                    r_num = int(round_folder.split("_")[1])
                    # If the round is older than actual_current_round - 2, clear its .pth files ONLY
                    if r_num < threshold_round:
                        for f in files:
                            if f.endswith(".pth"):
                                target_file = os.path.join(root, f)
                                os.remove(target_file)
                                deleted_count += 1
                                print(f"  🗑️ Deleted old round file: {target_file}")
                except ValueError:
                    pass # Ignore folders that are named "round_something" but aren't numbers
                    
        print(f"Cleanup complete. Removed {deleted_count} old .pth files.")
    else:
        print(f"Current round is {actual_current_round}. Threshold is {threshold_round}. No cleanup needed yet.")


def fedmc(models, fisher_paths, output_paths, norm_weights, client_ids, 
          prev_global_path="/workspace/work_dirs/fedmc_states/global_model.pth", 
          mask_dir="/workspace/work_dirs/fedmc_masks", 
          select_ratio=0.05, max_sparsity=0.5, no_reset_optimizer=False):
    """
    FedMC implementation (Information Content Model Customization). 
    Automatically discovers and freezes personalized subnetworks for each client 
    based on Fisher Information (parameter importance), then aggregates the shared parameters.
    """
    print(f"Running FedMC Aggregation (select_ratio={select_ratio}, max_sparsity={max_sparsity})...")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(os.path.dirname(prev_global_path), exist_ok=True)

    if not fisher_paths:
        raise ValueError("Error: 'fisher_paths' must be provided when running FedMC. Check your argparse inputs.")

    # # 1. Load Pre-Training Global Weights
    # if not os.path.exists(prev_global_path):
    #     print(f"No previous global model found at {prev_global_path}. Exiting FedMC since we need a baseline for aggregation. Please run one round of standard FedAvg first.")
    #     exit(1)

    # 1. Load / Recover Previous Global Weights
    if not os.path.exists(prev_global_path):
        import re

        print(
            f"No previous global model found at {prev_global_path}. "
            "Attempting reconstruction from previous round merged models and masks..."
        )

        # ---------------------------------------------------------
        # Determine current round from current output/model paths
        # ---------------------------------------------------------
        round_match = None

        for path in list(output_paths) + list(models):
            match = re.search(r'/round_(\d+)(?:/|$)', path)
            if match:
                round_match = match
                break

        if round_match is None:
            print("ERROR: Could not determine current round from paths:")
            for p in list(output_paths) + list(models):
                print(f"  - {p}")
            print("Expected a path containing '/round_X/'.")
            exit(1)

        current_round = int(round_match.group(1))
        previous_round = current_round - 1

        if previous_round < 1:
            print(
                f"ERROR: Current round is {current_round}; "
                "there is no previous round to recover."
            )
            exit(1)

        print(f"Current round: {current_round}")
        print(f"Trying to reconstruct global model from round {previous_round}.")

        # ---------------------------------------------------------
        # Get work directory from the current round path
        #
        # Example:
        # /workspace/work_dirs/round_11/merged_A.pth
        # ->
        # /workspace/work_dirs
        # ---------------------------------------------------------
        current_round_dir = None

        for path in list(output_paths) + list(models):
            if f"/round_{current_round}/" in path:
                prefix = path.split(f"/round_{current_round}/")[0]
                current_round_dir = os.path.join(
                    prefix,
                    f"round_{current_round}"
                )
                break

        if current_round_dir is None:
            print("ERROR: Could not determine current round directory.")
            exit(1)

        work_dir = os.path.dirname(current_round_dir)

        # ---------------------------------------------------------
        # Previous merged model paths
        # ModelA -> merged_A.pth
        # ---------------------------------------------------------
        previous_merged_paths = []

        for cid in client_ids:
            letter = cid.replace("Model", "")

            previous_merged_paths.append(
                os.path.join(
                    work_dir,
                    f"round_{previous_round}",
                    f"merged_{letter}.pth"
                )
            )

        # ---------------------------------------------------------
        # Check previous merged models exist
        # ---------------------------------------------------------
        missing_merged = [
            p for p in previous_merged_paths
            if not os.path.exists(p)
        ]

        if missing_merged:
            print(
                f"ERROR: Cannot reconstruct global model. "
                f"Missing merged checkpoints from round {previous_round}:"
            )

            for p in missing_merged:
                print(f"  - {p}")

            print("Exiting FedMC.")
            exit(1)

        print(
            f"Found all {len(previous_merged_paths)} merged models "
            f"from round {previous_round}."
        )

        # ---------------------------------------------------------
        # Check masks exist
        # ---------------------------------------------------------
        mask_paths = []
        for cid in client_ids:
            default_path = os.path.join(mask_dir, f"{cid}_mask.pth")
            round_0_path = os.path.join(mask_dir, "round_0", f"{cid}_mask.pt")
            
            # Use the .pt file in round_0 if it exists and the default doesn't
            if os.path.exists(round_0_path) and not os.path.exists(default_path):
                mask_paths.append(round_0_path)
            else:
                mask_paths.append(default_path)

        missing_masks = [
            p for p in mask_paths
            if not os.path.exists(p)
        ]

        # Special case:
        # If masks don't exist yet (e.g. recovering round 10 FedAvg
        # before the first FedMC round), all merged models should be
        # identical and any one of them is the global model.
        if missing_masks:

            print(
                "FedMC masks are missing. Checking whether previous "
                "merged models are identical (e.g. previous round was FedAvg)..."
            )

            ckpts = [
                torch.load(p, map_location="cpu")
                for p in previous_merged_paths
            ]

            states = [
                ckpt["state_dict"]
                for ckpt in ckpts
            ]

            identical = True

            for k in states[0]:
                if not states[0][k].is_floating_point():
                    continue

                for state_i in states[1:]:
                    if k not in state_i or not torch.equal(
                        states[0][k],
                        state_i[k]
                    ):
                        identical = False
                        break

                if not identical:
                    break

            if not identical:
                print(
                    "ERROR: Previous merged models differ, but the required "
                    "FedMC masks are missing:"
                )

                for p in missing_masks:
                    print(f"  - {p}")

                print(
                    "Cannot determine which parameters are global "
                    "and which are personalized."
                )
                print("Exiting FedMC.")
                exit(1)

            # Previous round was effectively a global/FedAvg round
            recovered_ckpt = ckpts[0]

            os.makedirs(
                os.path.dirname(prev_global_path),
                exist_ok=True
            )

            torch.save(
                recovered_ckpt,
                prev_global_path
            )

            print(
                f"Previous merged models are identical. "
                f"Successfully restored global model from round "
                f"{previous_round}:\n  {prev_global_path}"
            )

        else:
            # ---------------------------------------------------------
            # FedMC recovery using previous merged models + masks
            # ---------------------------------------------------------
            print(
                "All previous merged models and masks found. "
                "Reconstructing global model..."
            )

            previous_ckpts = [
                torch.load(p, map_location="cpu")
                for p in previous_merged_paths
            ]

            previous_states = [
                ckpt["state_dict"]
                for ckpt in previous_ckpts
            ]

            client_masks = [
                torch.load(p, map_location="cpu")
                for p in mask_paths
            ]

            # First previous merged checkpoint = checkpoint template
            recovered_ckpt = previous_ckpts[0]
            recovered_state = recovered_ckpt["state_dict"]

            valid_keys_recovery = [
                k for k, v in recovered_state.items()
                if v.is_floating_point()
                and "num_batches_tracked" not in k
            ]

            fully_personalized_params = 0
            total_valid_params = 0

            for k in valid_keys_recovery:

                recovered_sum = torch.zeros_like(
                    recovered_state[k]
                )

                recovered_count = torch.zeros_like(
                    recovered_state[k],
                    dtype=torch.float32
                )

                for state_i, client_mask, cid in zip(
                    previous_states,
                    client_masks,
                    client_ids
                ):
                    if k not in state_i:
                        print(
                            f"ERROR: '{k}' missing from previous "
                            f"merged checkpoint for {cid}."
                        )
                        exit(1)

                    if k not in client_mask:
                        print(
                            f"ERROR: '{k}' missing from mask for {cid}."
                        )
                        exit(1)

                    # mask == False -> this parameter was shared/global.
                    # Therefore merged_X contains the global value here.
                    active_mask = (~client_mask[k]).float()

                    recovered_sum += (
                        state_i[k] * active_mask
                    )

                    recovered_count += active_mask

                recoverable = recovered_count > 0
                unrecoverable = ~recoverable

                fully_personalized_params += (
                    unrecoverable.sum().item()
                )

                total_valid_params += recoverable.numel()

                # Recover global values wherever at least one client
                # still had that parameter globally shared.
                #
                # If ALL clients personalized the parameter, its global
                # value is irrelevant for future FedMC aggregation because
                # the masks are cumulative. Fill it with zero.
                recovered_state[k] = torch.where(
                    recoverable,
                    recovered_sum / recovered_count.clamp(min=1),
                    torch.zeros_like(recovered_state[k])
                )

            # ---------------------------------------------------------
            # Save reconstructed global model
            # ---------------------------------------------------------
            os.makedirs(
                os.path.dirname(prev_global_path),
                exist_ok=True
            )

            torch.save(
                recovered_ckpt,
                prev_global_path
            )

            pct = (
                100.0
                * fully_personalized_params
                / total_valid_params
                if total_valid_params > 0
                else 0.0
            )

            print(
                f"Successfully reconstructed global model from "
                f"round {previous_round}:\n"
                f"  {prev_global_path}"
            )

            print(
                f"Fully personalized parameters filled with zeros: "
                f"{fully_personalized_params:,}/{total_valid_params:,} "
                f"({pct:.4f}%)"
            )


    # Load recovered/existing global model normally
    prev_global_ckpt = torch.load(
        prev_global_path,
        map_location="cpu"
    )
    prev_state = prev_global_ckpt["state_dict"]
    
    # Identify valid floating-point keys
    valid_keys = [k for k, v in prev_state.items() if v.is_floating_point() and 'num_batches_tracked' not in k]
    total_params = sum(prev_state[k].numel() for k in valid_keys)
    print(f"Total valid parameters for FedMC: {total_params:,}")

    # ---------------------------------------------------------
    # Layer Mapping for Visualization (Created Once)
    # ---------------------------------------------------------
    mapping_file = os.path.join(mask_dir, "layer_mapping.pth")
    if not os.path.exists(mapping_file):
        layer_info = {}
        current_idx = 0
        for k in valid_keys:
            numel = prev_state[k].numel()
            layer_info[k] = {
                "shape": list(prev_state[k].shape),
                "numel": numel,
                "start_idx": current_idx,
                "end_idx": current_idx + numel
            }
            current_idx += numel
        torch.save(layer_info, mapping_file)
        print(f"Created layer mapping file at {mapping_file}")

    # ---------------------------------------------------------
    # Auto-Detect Current Round Directory
    # ---------------------------------------------------------
    existing_rounds = []
    for d in os.listdir(mask_dir):
        if d.startswith("round_") and os.path.isdir(os.path.join(mask_dir, d)):
            try:
                existing_rounds.append(int(d.split("_")[1]))
            except ValueError:
                pass
    current_round = max(existing_rounds) + 1 if existing_rounds else 0
    round_mask_dir = os.path.join(mask_dir, f"round_{current_round}")
    os.makedirs(round_mask_dir, exist_ok=True)
    print(f"Saving historical Fisher masks for round {current_round} to {round_mask_dir}")

    # ---------------------------------------------------------
    # Phase 1: Client Subnetwork Discovery using Fisher Information
    # ---------------------------------------------------------
    print("\nPhase 1: Discovering informative client subnetworks via Fisher Information...")
    for m_path, f_path, cid in zip(models, fisher_paths, client_ids):
        mask_path = os.path.join(mask_dir, f"{cid}_mask.pth")
        
        # Load or initialize client mask (0 = Global, 1 = Personalized)
        if os.path.exists(mask_path):
            client_mask = torch.load(mask_path)
        else:
            client_mask = {k: torch.zeros_like(prev_state[k], dtype=torch.bool) for k in valid_keys}

        ckpt_i = torch.load(m_path, map_location='cpu')
        state_i = ckpt_i['state_dict']
        
        # Load Fisher Information diagonal
        fisher_i = torch.load(f_path, map_location='cpu')
        if 'state_dict' in fisher_i: # Handle case if saved as a dict like checkpoints
            fisher_i = fisher_i['state_dict']
        
        
        # strip the wierd naming if it exists (e.g., 'module.layer1.weight' -> 'layer1.weight')
        fisher_i = {k.replace('module.', ''): v for k, v in fisher_i.items()}
        

        all_importance = []
        current_personalized = 0
        
        for k in valid_keys:
            if k in state_i and k in fisher_i:
                # Use Fisher Information as the importance metric (using abs as a safety net)
                importance = torch.abs(fisher_i[k])
                
                # Only evaluate parameters that are currently shared (mask == 0)
                global_mask = ~client_mask[k]
                all_importance.append(importance[global_mask].flatten())
                current_personalized += client_mask[k].sum().item()
                
        # Determine how many new parameters to select this round
        cat_importance = torch.cat(all_importance)
        k_to_select = int(total_params * select_ratio)
        max_allowed = int(total_params * max_sparsity)
        k_to_select = min(k_to_select, max_allowed - current_personalized)
        
        if k_to_select > 0 and len(cat_importance) > 0:
            k_to_select = min(k_to_select, len(cat_importance))
            
            # --- FISHER LOGGING ---
            avg_overall_fisher = cat_importance.mean().item()
            top_values = torch.topk(cat_importance, k_to_select).values
            threshold = top_values[-1].item()
            avg_selected_fisher = top_values.mean().item()
            
            print(f"\n[{cid}] FISHER INFORMATIVENESS:")
            print(f"  -> Avg Fisher (all shared params): {avg_overall_fisher:.6e}")
            print(f"  -> Avg Fisher (selected top-K):    {avg_selected_fisher:.6e}")
            print(f"  -> Fisher Threshold Cutoff:        {threshold:.6e}")
            
            # Update the mask permanently
            new_personalized = 0
            layer_counts = {}
            
            for k in valid_keys:
                if k in state_i and k in fisher_i:
                    importance = torch.abs(fisher_i[k])
                    
                    # Flip 0 to 1 if it exceeds the Fisher threshold and is currently 0
                    new_ones = (~client_mask[k]) & (importance >= threshold)
                    client_mask[k][new_ones] = True
                    
                    selected_in_layer = new_ones.sum().item()
                    new_personalized += selected_in_layer
                    
                    if selected_in_layer > 0:
                        layer_counts[k] = selected_in_layer
            
            # Print top 3 layers with the most highly informative parameters
            top_layers = sorted(layer_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  -> Most informative layers modified: {', '.join([f'{k} (+{v} params)' for k, v in top_layers])}")
            print(f"  -> Total Sparsity: {(current_personalized + new_personalized) / total_params * 100:.2f}% (+{new_personalized:,} new)")
        else:
            print(f"\n[{cid}] Reached max sparsity or no params to select. Total sparsity: {current_personalized / total_params * 100:.2f}%")
            
        torch.save(client_mask, mask_path)

        # Save historical visualization mask (0=Global, 1=Personal)
        vis_mask = {k: v.to(torch.int8) for k, v in client_mask.items()}
        hist_mask_path = os.path.join(round_mask_dir, f"{cid}_mask.pt")
        torch.save(vis_mask, hist_mask_path)

        del ckpt_i, fisher_i
        
    # ---------------------------------------------------------
    # Phase 2: Masked Server Aggregation
    # ---------------------------------------------------------
    print("\nPhase 2: Aggregating non-informative (shared) parameters on server...")
    running_sum = {k: torch.zeros_like(v) for k, v in prev_state.items() if k in valid_keys}
    presence_weights = {k: torch.zeros_like(v) for k, v in prev_state.items() if k in valid_keys}
    
    for m_path, cid, w_i in zip(models, client_ids, norm_weights):
        mask_path = os.path.join(mask_dir, f"{cid}_mask.pth")
        client_mask = torch.load(mask_path)
        
        ckpt_i = torch.load(m_path, map_location='cpu')
        state_i = ckpt_i['state_dict']
        
        for k in valid_keys:
            if k in state_i:
                # Active mask = 1 where parameter is shared (mask == 0)
                active_mask = (~client_mask[k]).float()
                running_sum[k] += state_i[k] * active_mask * w_i
                presence_weights[k] += active_mask * w_i
                
        del ckpt_i
        
    # Finalize averaged weights
    averaged_weights = {}
    for k in valid_keys:
        valid_mask = presence_weights[k] > 0
        averaged_weights[k] = torch.where(
            valid_mask,
            running_sum[k] / presence_weights[k].clamp(min=1e-9),
            prev_state[k] # Fallback if all clients personalized it
        )
        
    # Save the new global model
    for k in valid_keys:
        prev_global_ckpt['state_dict'][k] = averaged_weights[k]
    torch.save(prev_global_ckpt, prev_global_path)
    del prev_global_ckpt

    # ---------------------------------------------------------
    # Phase 3: Client Subnetwork Injection & Saving
    # ---------------------------------------------------------
    print("Phase 3: Injecting shared weights back into client models...")
    for in_path, out_path, cid in zip(models, output_paths, client_ids):
        ckpt = torch.load(in_path, map_location='cpu')
        state = ckpt['state_dict']
        
        mask_path = os.path.join(mask_dir, f"{cid}_mask.pth")
        client_mask = torch.load(mask_path)
        
        for k in valid_keys:
            if k in state:
                # Final weight = Mask * Local + (1 - Mask) * Global
                m = client_mask[k].float()
                state[k] = (m * state[k]) + ((1.0 - m) * averaged_weights[k])

        reset_optimizer_state(ckpt, no_reset_optimizer)
                        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(ckpt, out_path)
        print(f"Saved personalized FedMC model for {cid} to {out_path}")


def main():
    args = parse_args()
    
    if len(args.inputs) != len(args.outputs):
        raise ValueError("Number of input paths must match number of output paths.")
        
    model_paths = args.inputs
    output_paths = args.outputs
    cka_data_dirs = [
        args.data_dir_A,
        args.data_dir_B,
        args.data_dir_C,
        args.data_dir_D,
        args.data_dir_E,
    ]
    
    # Extract and normalize weights
    raw_weights = []
    for i in range(len(model_paths)):
        char = string.ascii_lowercase[i]
        w = getattr(args, f'weight_{char}', None)
        raw_weights.append(w if w is not None else 1.0)
        
    total_weight = sum(raw_weights)
    norm_weights = [w / total_weight for w in raw_weights]

    print(f"Loading {len(model_paths)} models for merge via {args.method.upper()}...")
    for i, (m_path, w) in enumerate(zip(model_paths, norm_weights)):
        print(f" Model {string.ascii_uppercase[i]}: {m_path} (normalized w={w:.4f})")
        
    # Route to the appropriate modular function
    if args.method == 'fedavg':
        print(f'Optimizer state will be reset: {not args.no_reset_optimizer}')
        fedavg(model_paths, output_paths, norm_weights, no_reset_optimizer=args.no_reset_optimizer)
        
    elif args.method == 'fedbn':
        if args.config is None:
            raise ValueError("You must provide a --config file to use FedBN so the model architecture can be built.")
            
        print(f"Building model from config: {args.config}")
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(['projects.mmdet3d_plugin.models.detectors.cmt'])
        
        from mmcv import Config
        cfg = Config.fromfile(args.config)
        if cfg.get('custom_imports', None):
            import_modules_from_strings(**cfg['custom_imports'])

        import importlib
        # import modules from plguin/xx, registry will be updated
        if hasattr(cfg, 'plugin'):
            if cfg.plugin:
                if hasattr(cfg, 'plugin_dir'):
                    plugin_dir = cfg.plugin_dir
                    _module_dir = os.path.dirname(plugin_dir)
                    _module_dir = _module_dir.split('/')
                    _module_path = _module_dir[0]

                    for m in _module_dir[1:]:
                        _module_path = _module_path + '.' + m
                    print(_module_path)
                    plg_lib = importlib.import_module(_module_path)
                else:
                    # import dir is the dirpath for the config file
                    _module_dir = os.path.dirname(args.config)
                    _module_dir = _module_dir.split('/')
                    _module_path = _module_dir[0]
                    for m in _module_dir[1:]:
                        _module_path = _module_path + '.' + m
                    print(_module_path)
                    plg_lib = importlib.import_module(_module_path)
                    
        plg_lib_base = importlib.import_module('mmdetection3d.mmdet3d')

        from mmdet3d.models import build_model
        model_instance = build_model(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))

        fedbn(model_paths, output_paths, norm_weights, model_instance, no_reset_optimizer=args.no_reset_optimizer)
    elif args.method == 'fedrep':
        fedrep(model_paths, output_paths, norm_weights, no_reset_optimizer=args.no_reset_optimizer)
        
    elif args.method == 'feddyn':
        feddyn(model_paths, output_paths, norm_weights, alpha=0.01, work_dir="work_dirs/feddyn_states", no_reset_optimizer=args.no_reset_optimizer)

    elif args.method == 'fedselect':
        client_ids = [f"Model{string.ascii_uppercase[i]}" for i in range(len(model_paths))]
        fedselect(
            models=model_paths, 
            output_paths=output_paths, 
            norm_weights=norm_weights, 
            client_ids=client_ids,
            prev_global_path="/workspace/work_dirs/fedselect_states/global_model.pth", 
            mask_dir="/workspace/work_dirs/fedselect_masks",
            select_ratio=args.select_ratio,
            max_sparsity=args.max_sparsity,
            no_reset_optimizer=args.no_reset_optimizer
        )
    elif args.method == 'fedselect_elastic':
        client_ids = [f"Model{string.ascii_uppercase[i]}" for i in range(len(model_paths))]
        fedselect_elastic(
            models=model_paths, 
            output_paths=output_paths, 
            norm_weights=norm_weights, 
            client_ids=client_ids,
            prev_global_path="/workspace/work_dirs/fedselect_states/global_model.pth", 
            mask_dir="/workspace/work_dirs/fedselect_masks",
            select_ratio=args.select_ratio,
            max_sparsity=args.max_sparsity,
            no_reset_optimizer=args.no_reset_optimizer
        )
    elif args.method == 'fedselect_fullelastic':
        client_ids = [f"Model{string.ascii_uppercase[i]}" for i in range(len(model_paths))]
        fedselect_fullelastic(
            models=model_paths, 
            output_paths=output_paths, 
            norm_weights=norm_weights, 
            client_ids=client_ids,
            prev_global_path="/workspace/work_dirs/fedselect_states/global_model.pth", 
            mask_dir="/workspace/work_dirs/fedselect_masks",
            select_ratio=args.select_ratio,
            no_reset_optimizer=args.no_reset_optimizer
        )
    elif args.method == 'fedselect_cka':
        client_ids = [f"Model{string.ascii_uppercase[i]}" for i in range(len(model_paths))]
        if len(model_paths) > len(cka_data_dirs) or any(
            data_dir is None for data_dir in cka_data_dirs[:len(model_paths)]
        ):
            raise ValueError("FedSelect CKA requires one dataset directory per client (A-E).")
        fedselect_cka(
            models=model_paths, 
            output_paths=output_paths, 
            norm_weights=norm_weights, 
            client_ids=client_ids,
            prev_global_path="/workspace/work_dirs/fedselect_states/global_model.pth", 
            mask_dir="/workspace/work_dirs/fedselect_masks",
            select_ratio=args.select_ratio,      # <-- Uses standard argument
            max_sparsity=args.max_sparsity,      # <-- Uses standard argument
            runner_path=args.runner_path,
            config=args.config,
            data_dirs=cka_data_dirs[:len(model_paths)],
            modality=args.modality,
            cka_samples=args.cka_samples,
            dynamic_sparsity=args.dynamic_sparsity,
            no_reset_optimizer=args.no_reset_optimizer
        )
    elif args.method == 'pcgrad':
            client_ids = [f"Model{string.ascii_uppercase[i]}" for i in range(len(model_paths))]
            PCGRAD(
                models=model_paths, 
                output_paths=output_paths, 
                norm_weights=norm_weights, 
                client_ids=client_ids,
                prev_global_path="/workspace/work_dirs/pcgrad_states/global_model.pth",
                no_reset_optimizer=args.no_reset_optimizer
            ) 
    elif args.method == 'fedmc':
        client_ids = [f"Model{string.ascii_uppercase[i]}" for i in range(len(model_paths))]
        fedmc(
            models=model_paths, 
            output_paths=output_paths, 
            norm_weights=norm_weights, 
            client_ids=client_ids,
            fisher_paths=args.fisher_paths,
            prev_global_path="/workspace/work_dirs/fedmc_states/global_model.pth", 
            mask_dir="/workspace/work_dirs/fedmc_masks",
            select_ratio=args.select_ratio,
            max_sparsity=args.max_sparsity,
            no_reset_optimizer=args.no_reset_optimizer
        )
    elif args.method == 'fedselect_cka_elastic':
        client_ids = [f"Model{string.ascii_uppercase[i]}" for i in range(len(model_paths))]
        if len(model_paths) > len(cka_data_dirs) or any(
            data_dir is None for data_dir in cka_data_dirs[:len(model_paths)]
        ):
            raise ValueError("FedSelect CKA elastic requires one dataset directory per client (A-E).")
        fedselect_cka_elastic(
            models=model_paths, 
            output_paths=output_paths, 
            norm_weights=norm_weights, 
            client_ids=client_ids,
            prev_global_path="/workspace/work_dirs/fedselect_states/global_model.pth", 
            mask_dir="/workspace/work_dirs/fedselect_masks",
            select_ratio=args.select_ratio,      # <-- Uses standard argument
            max_sparsity=args.max_sparsity,      # <-- Uses standard argument
            runner_path=args.runner_path,
            config=args.config,
            data_dirs=cka_data_dirs[:len(model_paths)],
            modality=args.modality,
            cka_samples=args.cka_samples,
            dynamic_sparsity=args.dynamic_sparsity,
            no_reset_optimizer=args.no_reset_optimizer
        )
if __name__ == '__main__':
    main()