"""
Base classes and utilities for model comparison metrics.
Provides modular architecture for implementing multiple similarity metrics.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any
from torch.nn.modules.batchnorm import _BatchNorm


class MetricCalculator(ABC):
    """Abstract base class for comparison metrics."""
    
    @abstractmethod
    def compute_layer_score(self, data_a: Any, data_b: Any) -> float:
        """
        Compute similarity score between two layer outputs/states.
        
        Args:
            data_a: Data from model A (format depends on metric)
            data_b: Data from model B (format depends on metric)
            
        Returns:
            float: Similarity score (typically 0-1)
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return human-readable name of the metric."""
        pass
    
    def get_filename(self) -> str:
        """Return filename-safe name of the metric."""
        return self.get_name().lower().replace(" ", "_").replace("/", "_")


class ActivationMetric(MetricCalculator):
    """Base class for metrics computed on layer activations."""
    
    @staticmethod
    def _normalize_activation(activation: torch.Tensor) -> torch.Tensor:
        """
        Normalize activation tensor to (N, D) shape for comparison.
        
        Handles:
        - 4D tensors (B, C, H, W) -> flatten spatial dims
        - 3D tensors (B, N, D) -> reshape
        - 2D tensors (N, D) -> keep as is
        - 1D tensors -> reshape
        """
        if activation is None or not torch.is_tensor(activation):
            return None
            
        x = activation
        
        if x.dim() == 4:
            b, c, h, w = x.shape
            x = x.permute(0, 2, 3, 1).reshape(-1, c)
        elif x.dim() == 3:
            b, n, d = x.shape
            x = x.reshape(-1, d)
        elif x.dim() == 2:
            x = x
        elif x.dim() == 1:
            x = x.reshape(-1, 1)
        else:
            x = x.reshape(x.shape[0], -1) if x.shape[0] > 0 else x.reshape(1, -1)
        
        return x
    
    @staticmethod
    def _match_sample_count(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Match number of samples in both tensors."""
        n = min(x.size(0), y.size(0))
        if n <= 1:
            return None, None
        return x[:n], y[:n]


class BatchNormMetric(MetricCalculator):
    """Base class for metrics computed on BatchNorm statistics."""
    
    @staticmethod
    def extract_bn_stats(module: _BatchNorm) -> Optional[Dict[str, torch.Tensor]]:
        """
        Extract running statistics from a BatchNorm module.
        
        Returns:
            Dict with 'mean' and 'var' (or None if not available)
        """
        if not isinstance(module, _BatchNorm):
            return None
        
        if not hasattr(module, 'running_mean') or not hasattr(module, 'running_var'):
            return None
        
        return {
            'mean': module.running_mean.detach().cpu().float(),
            'var': module.running_var.detach().cpu().float(),
        }
    
    @staticmethod
    def extract_bn_affine(module: _BatchNorm) -> Optional[Dict[str, torch.Tensor]]:
        """
        Extract affine parameters (scale/bias) from a BatchNorm module.
        
        Returns:
            Dict with 'weight' (gamma) and 'bias' (beta) (or None if not available)
        """
        if not isinstance(module, _BatchNorm):
            return None
        
        if not hasattr(module, 'weight') or not hasattr(module, 'bias'):
            return None
        
        return {
            'weight': module.weight.detach().cpu().float() if module.weight is not None else None,
            'bias': module.bias.detach().cpu().float() if module.bias is not None else None,
        }


class ComparisonResult:
    """Container for comparison results across metrics and layers."""
    
    def __init__(self):
        # Dict[metric_name] -> Dict[layer_name] -> float
        self.scores: Dict[str, Dict[str, float]] = {}
        self.metric_names: list = []
    
    def add_metric(self, metric_name: str, layer_scores: Dict[str, float]):
        """Add results for a metric."""
        self.scores[metric_name] = layer_scores
        if metric_name not in self.metric_names:
            self.metric_names.append(metric_name)
    
    def get_layers_ordered(self) -> list:
        """Get all layers in consistent order (alphabetical)."""
        all_layers = set()
        for metric_scores in self.scores.values():
            all_layers.update(metric_scores.keys())
        return sorted(list(all_layers))
    
    def get_matrix(self, comparisons: list, layer_order: list) -> Dict[str, np.ndarray]:
        """
        Create matrices for visualization (one per metric).
        
        Args:
            comparisons: List of comparison titles
            layer_order: Ordered list of layers
            
        Returns:
            Dict[metric_name] -> numpy array (num_comparisons, num_layers)
        """
        matrices = {}
        for metric_name in self.metric_names:
            matrix = np.zeros((len(comparisons), len(layer_order)))
            
            for i, comp_title in enumerate(comparisons):
                # This will be populated by caller
                matrices[metric_name] = matrix
        
        return matrices