"""
Core comparison engine for multi-metric model comparison.
Handles the workflow of comparing two models across multiple metrics.
"""

import os
import shutil
import time
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable
from tqdm import tqdm
from torch.nn.modules.batchnorm import _BatchNorm

from metrics_base import MetricCalculator
from metrics import CKAMetric, W2BNStatsMetric, EffectiveScaleBiasMetric
from comparison_utils import (
    ForwardHook, load_model_and_pipeline, register_hooks,
    process_single_sample, get_bn_module_refs, default_hook_filter,
    sort_layers_by_network_order
)


class ModelComparator:
    """
    Orchestrates comparison of two models across multiple metrics.
    
    Workflow:
    1. For activation metrics (CKA): Cache model A activations, then stream model B
    2. For BN metrics (W2, Effective scale/bias): Extract and compare BN modules
    """
    
    def __init__(
        self,
        config_path: str,
        sample_files: List[str],
        temp_dir: str = "./temp_comparisons",
        verbose: bool = False
    ):
        """
        Initialize comparator.
        
        Args:
            config_path: Path to model config
            sample_files: List of sample files for inference
            temp_dir: Directory for temporary cached data
            verbose: Whether to enable verbose logging
        """
        self.config_path = config_path
        self.sample_files = sample_files
        self.temp_dir = temp_dir
        self.verbose = verbose
        self.bn_module_cache = {}  # Cache BN modules for efficient reuse

    def _log(self, message: str):
        """Print only when verbose logging is enabled."""
        if self.verbose:
            print(message)
    
    def _setup_temp_dir(self):
        """Create fresh temp directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir)
    
    def _cleanup_temp_dir(self):
        """Clean up temp directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def compare_activation_metric(
        self,
        model_a_path: str,
        model_b_path: str,
        metric: MetricCalculator,
        hook_filter: Optional[Callable] = None
    ) -> Dict[str, float]:
        """
        Compare two models using an activation-based metric (e.g., CKA).
        
        Workflow:
        1. Load Model A, collect activations from all samples, cache to disk
        2. Load Model B, stream through samples, compute metric on-the-fly
        
        Args:
            model_a_path: Path to checkpoint for model A
            model_b_path: Path to checkpoint for model B
            metric: Metric calculator instance
            hook_filter: Function to select layers to compare
            
        Returns:
            Dict mapping layer names to metric scores
        """
        if hook_filter is None:
            from comparison_utils import default_hook_filter
            hook_filter = default_hook_filter
        
        self._setup_temp_dir()

        start_time = time.time()
        print(f"Starting activation metric comparison at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        try:
            # Phase 1: Cache Model A activations
            self._log(f"  [Phase 1] Caching activations for Model A ({len(self.sample_files)} samples)...")
            model_a, pipeline_a = load_model_and_pipeline(self.config_path, model_a_path)
            hooks_a, handles_a = register_hooks(model_a, hook_filter)
            
            for i, sample in enumerate(tqdm(self.sample_files, desc="Model A")):
                acts = process_single_sample(model_a, pipeline_a, sample, hooks_a)
                torch.save(acts, os.path.join(self.temp_dir, f"{i}.pt"))
                
                # Debug first sample
                if i == 0:
                    self._log(f"    Model A sample 0: captured {len(acts)} activations")
                    for name, act in list(acts.items())[:3]:
                        if act is not None:
                            if isinstance(act, dict) and act.get("is_sparse"):
                                self._log(f"      {name}: SPARSE dict {act['features'].shape}, CPU (cached)")
                            else:
                                self._log(f"      {name}: {act.shape}, {act.device}")
                        else:
                            self._log(f"      {name}: None")
            
            for h in handles_a:
                h.remove()
            del model_a
            torch.cuda.empty_cache()

            print(f"Phase 1 completed in {time.time() - start_time:.2f} seconds.")
            
            # Phase 2: Stream Model B and compute metric
            self._log(f"  [Phase 2] Computing {metric.get_name()} with Model B...")
            model_b, pipeline_b = load_model_and_pipeline(self.config_path, model_b_path)
            hooks_b, handles_b = register_hooks(model_b, hook_filter)
            
            cumulative_scores = {}
            counts = {}
            
            for i, sample in enumerate(tqdm(self.sample_files, desc="Model B")):
                acts_b = process_single_sample(model_b, pipeline_b, sample, hooks_b)
                
                # Debug first sample
                if i == 0:
                    self._log(f"    Model B sample 0: captured {len(acts_b)} activations")
                    for name, act in list(acts_b.items())[:3]:
                        if act is not None:
                            if isinstance(act, dict) and act.get("is_sparse"):
                                self._log(f"      {name}: SPARSE dict {act['features'].shape}, CPU (cached)")
                            else:
                                self._log(f"      {name}: {act.shape}, {act.device}")
                        else:
                            self._log(f"      {name}: None")
                
                try:
                    acts_a = torch.load(os.path.join(self.temp_dir, f"{i}.pt"))
                except FileNotFoundError:
                    continue
                
                # Compute metric for common layers
                common_layers = set(acts_a.keys()) & set(acts_b.keys())
                
                # Debug common layers
                if i == 0:
                    self._log(f"    Sample 0: {len(common_layers)} common layers out of A={len(acts_a)} B={len(acts_b)}")
                    if len(common_layers) > 0:
                        self._log(f"    Common layer examples: {list(common_layers)[:3]}")
                
                for layer in common_layers:
                    score = metric.compute_layer_score(acts_a[layer], acts_b[layer])
                    
                    if layer not in cumulative_scores:
                        cumulative_scores[layer] = 0.0
                        counts[layer] = 0
                    
                    cumulative_scores[layer] += score
                    counts[layer] += 1
            
            for h in handles_b:
                h.remove()
            del model_b
            torch.cuda.empty_cache()
            
            # Average the scores
            avg_scores = {k: v / counts[k] for k, v in cumulative_scores.items()}
            print(f"Phase 2 completed in {time.time() - start_time:.2f} seconds.")
            return avg_scores
        
        finally:
            self._cleanup_temp_dir()
    
    def compare_bn_metric(
        self,
        model_a_path: str,
        model_b_path: str,
        metric: MetricCalculator,
        sample_path_for_init: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Compare two models using a BatchNorm metric (e.g., W2, Effective scale/bias).
        
        These metrics don't require inference, only module extraction.
        However, we run inference on one sample to ensure BN modules are properly initialized.
        
        Args:
            model_a_path: Path to checkpoint for model A
            model_b_path: Path to checkpoint for model B
            metric: Metric calculator instance
            sample_path_for_init: One sample file to run through for BN initialization
            
        Returns:
            Dict mapping layer names to metric scores
        """
        self._log(f"  Computing {metric.get_name()} (extracting BN modules)...")
        
        # Load both models
        model_a, pipeline_a = load_model_and_pipeline(self.config_path, model_a_path)
        model_b, pipeline_b = load_model_and_pipeline(self.config_path, model_b_path)
        
        # Get BN module references
        bn_modules_a = self._get_bn_modules(model_a)
        bn_modules_b = self._get_bn_modules(model_b)
        
        # Run one sample through each model to ensure BN statistics are valid
        if sample_path_for_init and len(self.sample_files) > 0:
            sample = self.sample_files[0]
            with torch.no_grad():
                from comparison_utils import process_single_sample, register_hooks, default_hook_filter
                
                hooks_a, handles_a = register_hooks(model_a, default_hook_filter)
                process_single_sample(model_a, pipeline_a, sample, hooks_a)
                for h in handles_a:
                    h.remove()
                
                hooks_b, handles_b = register_hooks(model_b, default_hook_filter)
                process_single_sample(model_b, pipeline_b, sample, hooks_b)
                for h in handles_b:
                    h.remove()
        
        # Compute metric for each BN layer
        scores = {}
        common_bn_layers = set(bn_modules_a.keys()) & set(bn_modules_b.keys())
        
        for layer_name in common_bn_layers:
            self._log(f"    Comparing BN Layer: {layer_name} ...")
            score = metric.compute_layer_score(
                bn_modules_a[layer_name],
                bn_modules_b[layer_name]
            )
            scores[layer_name] = score
        
        del model_a, model_b
        torch.cuda.empty_cache()
        
        return scores
    
    def _get_bn_modules(self, model: nn.Module) -> Dict[str, _BatchNorm]:
        """Extract all BatchNorm modules from model."""
        bn_modules = {}
        
        for name, module in model.named_modules():
            if isinstance(module, _BatchNorm):
                bn_modules[name] = module
        
        return bn_modules
    
    def run_full_comparison(
        self,
        model_a_path: str,
        model_b_path: str,
        metrics: List[MetricCalculator],
        hook_filter: Optional[Callable] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Run full comparison with multiple metrics.
        
        Args:
            model_a_path: Path to checkpoint for model A
            model_b_path: Path to checkpoint for model B
            metrics: List of metric calculators
            hook_filter: Function to select layers
            
        Returns:
            Dict[metric_name] -> Dict[layer_name] -> score
        """
        results = {}
        
        for metric in metrics:
            metric_name = metric.get_name()
            self._log(f"\n[{metric_name}]")
            
            if isinstance(metric, CKAMetric):
                # Activation-based metric
                scores = self.compare_activation_metric(
                    model_a_path, model_b_path, metric, hook_filter
                )
            else:
                # BN-based metric
                scores = self.compare_bn_metric(
                    model_a_path, model_b_path, metric
                )
            
            results[metric_name] = scores
            self._log(f"  Compared {len(scores)} layers")
        
        return results
    
