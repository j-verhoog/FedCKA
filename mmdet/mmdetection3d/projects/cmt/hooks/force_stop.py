from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class ForceStopHook(Hook):
    """Stops training at a specific epoch, regardless of total_epochs."""
    def __init__(self, stop_epoch):
        self.stop_epoch = stop_epoch

    def after_train_epoch(self, runner):
        # runner.epoch is 0-indexed. If we want to stop after epoch 1,
        # runner.epoch will be 0.
        if (runner.epoch + 1) >= self.stop_epoch:
            runner.logger.info(f"ForceStopHook: Reached target epoch {self.stop_epoch}. Stopping.")
            # Trick the runner into thinking it's done
            runner._max_epochs = runner.epoch + 1