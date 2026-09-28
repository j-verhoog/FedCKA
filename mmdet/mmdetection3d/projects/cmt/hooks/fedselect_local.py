import os
import torch
from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class FedSelectLocalTrainingHook(Hook):
    """
    Applies element-wise gradient scaling for FedSelect during local client training.

    Personalized parameters (mask=1) receive normal gradients.
    Global parameters (mask=0) receive heavily scaled-down gradients (or 0)
    to preserve global knowledge.

    FedSelect uses LocalAlt; this implementation is LocalSim.
    The FedSelect paper on which LocalAlt/LocalSim is based is:
    [34] K. Pillutla, K. Malik, A. Mohamed, M. Rabbat, M. Sanjabi,
    and L. Xiao. Federated learning with partial model personalization. In International Conference on Machine Learning,
    2022.
    """
    def __init__(self, mask_path, global_lr_scale=0.01, personal_lr_scale=1.0):
        """
        Args:
            mask_path (str): Path to the client's saved binary mask dictionary.
            global_lr_scale (float): Multiplier for global parameter gradients (default 0.01 per paper).
            personal_lr_scale (float): Multiplier for personalized parameter gradients.
        """
        self.mask_path = mask_path
        self.global_lr_scale = global_lr_scale
        self.personal_lr_scale = personal_lr_scale
        self.client_mask = None
        self._hook_handles = []
        self.trigger_path = '/workspace/work_dirs/fedselect_states/global_model.pth'

    def before_train(self, runner):
        """
        Loads the mask and registers PyTorch backward hooks on the parameters
        before training begins.
        """
        # if no global model, do nothing
        if not os.path.exists(self.trigger_path):
            runner.logger.info(f"FedAvg phase detected (trigger {self.trigger_path} not found). FedSelect hooks disabled.")
            return

        runner.logger.info(f"--- Initializing FedSelect Hook ---")
        runner.logger.info(f"Loading mask from {self.mask_path}")
        
        if not os.path.exists(self.mask_path):
            runner.logger.warning(f"No mask found at {self.mask_path}. "
                                  f"Assuming round 1 (all params global).")
            self.client_mask = {} # Empty mask means everything defaults to global
        else:
            self.client_mask = torch.load(self.mask_path, map_location='cpu')

        hooked_params = 0
        total_params = 0

        # Iterate through the model's parameters and attach the gradient modifier
        for name, param in runner.model.named_parameters():
            if not param.requires_grad:
                continue
            
            total_params += 1
            
            # Strip 'module.' prefix if wrapped in DistributedDataParallel
            clean_name = name.replace('module.', '')
            
            # Create a localized closure for the hook
            def make_grad_hook(param_name_clean):
                def grad_hook(grad):
                    # If parameter isn't in mask yet, it is completely global
                    if param_name_clean not in self.client_mask:
                        return grad * self.global_lr_scale

                    # Load the mask and move it to the parameter's device (GPU)
                    mask = self.client_mask[param_name_clean].to(grad.device)
                    
                    # mask == 1 (Personalized) -> scale by personal_lr_scale
                    # mask == 0 (Global) -> scale by global_lr_scale
                    scale_matrix = torch.where(mask, 
                                               torch.tensor(self.personal_lr_scale, device=grad.device), 
                                               torch.tensor(self.global_lr_scale, device=grad.device))
                    
                    # Return the modified gradient
                    return grad * scale_matrix
                return grad_hook

            # Register the hook and save the handle so we can remove it later
            handle = param.register_hook(make_grad_hook(clean_name))
            self._hook_handles.append(handle)
            hooked_params += 1

        runner.logger.info(f"Registered FedSelect element-wise gradient hooks on {hooked_params}/{total_params} trainable layers.")
        runner.logger.info(f"Global LR Scale: {self.global_lr_scale} | Personal LR Scale: {self.personal_lr_scale}")

    def after_train(self, runner):
        """
        Clean up the hooks after training to prevent memory leaks if the model 
        is kept in memory for evaluation.
        """
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        runner.logger.info("Removed FedSelect gradient hooks.")