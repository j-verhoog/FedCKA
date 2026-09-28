import torch

def compare_model_weights(path1, path2, tolerance=1e-6):
    print(f"Loading checkpoints...\n1: {path1}\n2: {path2}\n")
    
    # Load checkpoints to CPU
    ckpt1 = torch.load(path1, map_location='cpu')
    ckpt2 = torch.load(path2, map_location='cpu')
    
    # Extract state_dicts (handles both full checkpoints and raw state_dicts)
    sd1 = ckpt1.get('state_dict', ckpt1) if isinstance(ckpt1, dict) else ckpt1
    sd2 = ckpt2.get('state_dict', ckpt2) if isinstance(ckpt2, dict) else ckpt2
    
    merged_same = []
    local_different = []
    
    for key, tensor1 in sd1.items():
        if key not in sd2:
            print(f"Warning: Layer '{key}' not found in the second model.")
            continue
            
        tensor2 = sd2[key]
        
        # Skip non-tensors or non-floating point tracking variables (like num_batches_tracked)
        if not torch.is_tensor(tensor1) or not torch.is_tensor(tensor2):
            continue
        if not tensor1.is_floating_point():
            # Use strict equality for integers/booleans
            is_identical = torch.equal(tensor1, tensor2)
        else:
            # Use allclose for floating point weights
            is_identical = torch.allclose(tensor1, tensor2, atol=tolerance)
            
        if is_identical:
            merged_same.append(key)
        else:
            local_different.append(key)

    # --- Print Results ---
    print("-" * 50)
    print(f"✅ IDENTICAL LAYERS (Merged successfully): {len(merged_same)}")
    # Uncomment the next line to print all identical layer names
    # print("\n".join(merged_same)) 
    
    print("-" * 50)
    print(f"❌ DIFFERENT LAYERS (Kept local/personalized): {len(local_different)}")
    for layer in local_different:
        print(f"  - {layer}")
    print("-" * 50)

if __name__ == "__main__":
    # Replace these paths with the paths to your specific .pth files
    model_a_path = "/YOUR_HOMEDIR_PATH_HERE/Desktop/FedBN_modelA_merged_e20.pth"
    model_b_path = "/YOUR_HOMEDIR_PATH_HERE/Desktop/FedBN_modelB_merged_e20.pth"

    compare_model_weights(model_a_path, model_b_path)