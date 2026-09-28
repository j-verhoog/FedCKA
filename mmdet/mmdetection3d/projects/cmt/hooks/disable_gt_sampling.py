from mmcv.runner import HOOKS, Hook

class SamplingDisablerProxy:
    """
    A wrapper that sits around the real DB Sampler.
    It passes all attribute access to the real sampler (so nothing breaks),
    but it intercepts 'sample_all' to return None (disabling sampling).
    """
    def __init__(self, original_sampler):
        self._original_sampler = original_sampler

    def sample_all(self, *args, **kwargs):
        # We return None. This tells UnifiedObjectSample (Line 126) 
        # that no samples were found, so it skips the mixing logic cleanly.
        return None

    def __getattr__(self, name):
        # If code asks for 'db_infos', 'data_root', etc., 
        # we fetch it from the real sampler.
        return getattr(self._original_sampler, name)


@HOOKS.register_module()
class DisableGTImportanceHook(Hook):
    def __init__(self, disable_after_epoch=15):
        self.disable_after_epoch = disable_after_epoch
        self._has_disabled = False

    def before_train_epoch(self, runner):
        # Check if we should disable
        if runner.epoch >= self.disable_after_epoch and not self._has_disabled:
            
            # 1. Locate the dataset
            dataset = runner.data_loader.dataset
            # Handle CBGS/Dataset wrapping
            if hasattr(dataset, 'dataset'):
                real_dataset = dataset.dataset
            else:
                real_dataset = dataset

            # 2. Locate the Transform
            found = False
            if hasattr(real_dataset, 'pipeline'):
                for transform in real_dataset.pipeline.transforms:
                    if 'UnifiedObjectSample' in type(transform).__name__:
                        
                        # 3. Apply the Proxy
                        # We do not destroy the object. We wrap it.
                        if not isinstance(transform.db_sampler, SamplingDisablerProxy):
                            runner.logger.info(f"Epoch {runner.epoch}: wrapping db_sampler in DisablerProxy...")
                            transform.db_sampler = SamplingDisablerProxy(transform.db_sampler)
                            found = True
                        
                        # Optimization: toggle boolean flag if it exists, though Proxy handles the logic.
                        if hasattr(transform, 'sample_2d'):
                            transform.sample_2d = False

            if found:
                runner.logger.info(f"Epoch {runner.epoch}: GT Sampling has been SAFELY disabled via Proxy.")
                self._has_disabled = True
            else:
                runner.logger.warning(f"Epoch {runner.epoch}: Could not find UnifiedObjectSample to disable!")