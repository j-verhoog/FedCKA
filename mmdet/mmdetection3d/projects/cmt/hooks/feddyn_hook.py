import os
import torch
from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class FedDynHook(Hook):
    def __init__(self, client_id, start_epoch=10, alpha=0.01, work_dir='./feddyn_states'):
        self.client_id = client_id
        self.alpha = alpha # this influences how hard the local models are pulled towards the global model.
        self.work_dir = work_dir
        self.start_epoch = start_epoch 
        self.global_weights = {}
        self.h_states = {}
        self.handles = []

    def before_run(self, runner):
        # Only initialize if we have passed the warm-up phase
        if runner.epoch < self.start_epoch:
            runner.logger.info(f"FedDyn warm-up phase. Skipping penalty for Epoch {runner.epoch}.")
            return
        
        runner.logger.info(f"--- Initializing FedDyn Hooks for {self.client_id} ---")
        
        # 1. The model just loaded weights from the server. Clone them to act as frozen global weights.
        for name, param in runner.model.named_parameters():
            self.global_weights[name] = param.clone().detach().cpu()

        # 2. Load the historical state 'h' for this client, or init to zeros for Round 1
        os.makedirs(self.work_dir, exist_ok=True)
        h_path = os.path.join(self.work_dir, f"{self.client_id}_h_state.pth")
        
        if os.path.exists(h_path):
            self.h_states = torch.load(h_path)
            runner.logger.info(f"Loaded existing FedDyn h_state from {h_path}")
        else:
            for name, param in runner.model.named_parameters():
                self.h_states[name] = torch.zeros_like(param.data, device='cpu')
            runner.logger.info(f"Initialized new FedDyn h_state for {self.client_id}")

        # 3. Attach PyTorch backward hooks to intercept gradients on the fly
        for name, param in runner.model.named_parameters():
            if param.requires_grad:
                handle = param.register_hook(self._get_hook(name, param))
                self.handles.append(handle)

    def _get_hook(self, param_name, local_param):
        # Basic string matching for BatchNorm (adjust if your model uses different BN names)
        is_bn = '.bn' in param_name or '.norm' in param_name or 'bn.' in param_name

        def hook_fn(grad):
            if grad is None: return grad

            # If it's a BN layer, skip the FedDyn penalty.
            # These layers are relatively small and update fiercely so feddyn de-stabilizes them. We let them update freely.
            if is_bn:
                return grad
            
            # Move frozen global weights and h_state to the current GPU layer-by-layer to save VRAM
            global_w = self.global_weights[param_name].to(grad.device)
            h_state = self.h_states[param_name].to(grad.device)

            # Inside your FedDyn hook's _get_hook function:
            layer_alpha = self.alpha

            if 'img_backbone' in param_name:
                layer_alpha = self.alpha * 0.01     # very deep network, so we scale down the alpha to avoid over-regularization
            elif 'img_neck' in param_name:
                layer_alpha = self.alpha * 0.1 
            
            # Apply FedDyn math: grad = grad - h + alpha * (local_w - global_w)
            feddyn_penalty = -h_state + layer_alpha * (local_param.detach() - global_w)
            return grad + feddyn_penalty
            
        return hook_fn

    def after_run(self, runner):
        # Do not save states or update H if we are still warming up
        if runner.epoch < self.start_epoch:
            return
        
        runner.logger.info(f"--- Updating and Saving FedDyn States for {self.client_id} ---")
        h_path = os.path.join(self.work_dir, f"{self.client_id}_h_state.pth")
        
        # Update h_states: h_new = h_old - alpha * (w_final - w_global)
        for name, param in runner.model.named_parameters():
            if name in self.h_states:
                global_w = self.global_weights[name].to(param.device)
                h_old = self.h_states[name].to(param.device)
                
                # Apply the same layer-wise scaling to the H-update
                layer_alpha = self.alpha
                if 'img_backbone' in name:
                    layer_alpha = self.alpha * 0.01
                elif 'img_neck' in name:
                    layer_alpha = self.alpha * 0.1

                new_h = h_old - layer_alpha * (param.detach() - global_w)
                self.h_states[name] = new_h.cpu()

        torch.save(self.h_states, h_path)

        # Clean up hooks to prevent memory leaks across rounds
        for handle in self.handles:
            handle.remove()
        self.handles = []