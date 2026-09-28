"""
Visualization utilities for model comparison results.
Generates heatmaps for each metric with consistent styling.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from comparison_utils import sort_layers_by_network_order


class ComparisonVisualizer:
    """
    Handles visualization of multi-metric comparison results.
    """
    
    # Color schemes for different metrics
    COLORMAPS = {
        'CKA': 'RdYlGn_r',
        'W2 BN Stats': 'RdYlGn_r',  # Red (high) to Green (low)
        'Effective Scale/Bias': 'RdYlGn_r',
        'Bhattacharyya Coeff': 'RdYlGn_r',
    }
    
    # V-min and max for consistent scaling
    VMIN_MAX = {
        'CKA': (0.0, 1.0),
        'W2 BN Stats': (0.0, 1.0),
        'Effective Scale/Bias': (0.0, 1.0),
        'Bhattacharyya Coeff': (0.0, 1.0),
    }
    
    @staticmethod
    def _get_comparison_titles(comparisons: List[Tuple[str, str, str]]) -> List[str]:
        """Extract comparison titles from (title, model1, model2) tuples."""
        return [c[0] for c in comparisons]
    
    @staticmethod
    def _build_matrix(
        results: Dict[str, Dict[str, float]],
        metric_name: str,
        comparisons: List[Tuple[str, str, str]],
        layer_order: List[str]
    ) -> np.ndarray:
        """
        Build matrix for a specific metric.
        
        Args:
            results: Dict[comparison_title] -> Dict[layer_name] -> score
            metric_name: Name of metric to extract
            comparisons: List of (title, model1, model2) tuples
            layer_order: Ordered list of layer names
            
        Returns:
            Matrix of shape (num_comparisons, num_layers)
        """
        # Initialize with NaN instead of 0.0
        matrix = np.full((len(comparisons), len(layer_order)), np.nan)
        
        for i, (comp_title, _, _) in enumerate(comparisons):
            if comp_title in results and metric_name in results[comp_title]:
                metric_scores = results[comp_title][metric_name]
                
                # --- FIX: INDENT THIS LOOP INSIDE THE IF BLOCK ---
                for j, layer in enumerate(layer_order):
                    if layer in metric_scores:
                        matrix[i, j] = metric_scores[layer]

        return matrix
    
    @staticmethod
    def plot_single_metric(
        matrix: np.ndarray,
        comparisons: List[Tuple[str, str, str]],
        layer_order: List[str],
        metric_name: str,
        figsize: Tuple[int, int] = (20, 6),
        output_path: str = None
    ) -> str:
        """
        Plot heatmap for a single metric.
        
        Args:
            matrix: Comparison matrix (num_comparisons, num_layers)
            comparisons: List of (title, model1, model2) tuples
            layer_order: Ordered list of layer names
            metric_name: Name of metric (for title and coloring)
            figsize: Figure size
            output_path: If provided, save to this path
            
        Returns:
            Output filename
        """
        comp_titles = [c[0] for c in comparisons]
        
        # Get colormap and vmin/vmax
        cmap = ComparisonVisualizer.COLORMAPS.get(metric_name, 'viridis')
        vmin, vmax = ComparisonVisualizer.VMIN_MAX.get(metric_name, (0.0, 1.0))
        
        # Get colormap and set the 'bad' value color to grey
        base_cmap = plt.get_cmap(ComparisonVisualizer.COLORMAPS.get(metric_name, 'viridis')).copy()
        base_cmap.set_bad(color='lightgrey') # This makes NaNs grey
        
        # CREATE THE MASK: True where data is NaN
        mask = np.isnan(matrix)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # --- CRITICAL FIX START ---
        # Paint the background grey. 
        # The mask makes the cells transparent, revealing this color.
        ax.set_facecolor('lightgrey')
        # --- CRITICAL FIX END ---
        sns.heatmap(
            matrix,
            xticklabels=layer_order,
            yticklabels=comp_titles,
            annot=False,
            cmap=base_cmap, # test with new suggestion for grey nan
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            cbar_kws={'label': metric_name},
            mask=mask
        )
        
        plt.title(f"{metric_name} Comparison Heatmap")
        plt.xticks(rotation=45, ha='right', fontsize=4)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        
        # Generate output filename if not provided
        if output_path is None:
            metric_safe_name = metric_name.lower().replace(" ", "_").replace("/", "_")
            output_path = f"{metric_safe_name}_heatmap.png"
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    @staticmethod
    def plot_all_metrics(
        results: Dict[str, Dict[str, Dict[str, float]]],
        comparisons: List[Tuple[str, str, str]],
        metric_names: List[str],
        figsize_per_metric: Tuple[int, int] = (20, 6),
        output_dir: str = ".",
    ) -> Dict[str, str]:
        """
        Generate heatmaps for all metrics.
        
        Args:
            results: Dict[comparison_title] -> Dict[metric_name] -> Dict[layer_name] -> score
            comparisons: List of (title, model1, model2) tuples
            metric_names: List of metric names to plot
            figsize_per_metric: Figure size per metric
            output_dir: Directory to save plots
            
        Returns:
            Dict mapping metric names to output filenames
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        output_files = {}
        
        # Get all layers from all comparisons and metrics
        all_layers = set()
        for comp_results in results.values():
            for metric_scores in comp_results.values():
                all_layers.update(metric_scores.keys())
        
        layer_order = sort_layers_by_network_order(list(all_layers))
        
        # Plot each metric
        for metric_name in metric_names:
            print(f"Plotting {metric_name}...")
            
            matrix = ComparisonVisualizer._build_matrix(
                results, metric_name, comparisons, layer_order
            )
            
            # --- DEBUG PRINT ---
            # Use this to verify your matrix actually has NaNs
            nans = np.isnan(matrix).sum()
            zeros = (matrix == 0).sum()
            total = matrix.size
            print(f"  Matrix Stats -> NaNs: {nans}, Zeros: {zeros}, Total: {total}")
            # -------------------

            output_path = os.path.join(
                output_dir,
                f"{metric_name.lower().replace(' ', '_').replace('/', '_')}_heatmap.png"
            )
            
            output_file = ComparisonVisualizer.plot_single_metric(
                matrix, comparisons, layer_order, metric_name,
                figsize=figsize_per_metric,
                output_path=output_path
            )
            
            output_files[metric_name] = output_file
            print(f"  Saved: {output_file}")
        
        return output_files
    
    @staticmethod
    def plot_consolidated(
        results: Dict[str, Dict[str, Dict[str, float]]],
        comparisons: List[Tuple[str, str, str]],
        layer_order: List[str],
        output_path: str,
        run_name: str = None
    ):
        """
        Plots 4 metrics vertically with NO gaps and simplified labels.
        """
        metric_order = [
            'Effective Scale/Bias',
            'Bhattacharyya Coeff',
            'W2 BN Stats',
            'CKA'
        ]
        
        # 1. Setup Figure
        # hspace=0 removes vertical empty space between plots
        fig, axes = plt.subplots(
            nrows=4, 
            ncols=1, 
            figsize=(24, 10), 
            sharex=True,      
            gridspec_kw={'hspace': 0} 
        )
        
        # Shared colorbar axis
        cbar_ax = fig.add_axes([.91, .3, .015, .4]) 

        for i, metric_name in enumerate(metric_order):
            ax = axes[i]
            
            matrix = ComparisonVisualizer._build_matrix(
                results, metric_name, comparisons, layer_order
            )
            
            cmap = ComparisonVisualizer.COLORMAPS.get(metric_name, 'RdYlGn')
            base_cmap = plt.get_cmap(cmap).copy()
            base_cmap.set_bad(color='lightgrey')
            mask = np.isnan(matrix)
            
            # 2. Heatmap
            sns.heatmap(
                matrix,
                xticklabels=layer_order,
                yticklabels=[], # REMOVE "Run Label" from side
                annot=False,
                cmap=base_cmap,
                vmin=0.0, vmax=1.0,
                ax=ax,
                mask=mask,
                cbar=(i == 0),
                cbar_ax=cbar_ax if i == 0 else None
            )
            
            # 3. Styling
            ax.set_facecolor('lightgrey')
            
            # Metric Name on LEFT (Y-Axis Label)
            # We rotate it 0 degrees (horizontal) or 90 (vertical) depending on preference.
            # Vertical (standard Y-label) usually fits better if names are long.
            ax.set_ylabel(metric_name, fontsize=11, fontweight='bold', labelpad=10)
            
            # Clear individual subplot titles
            ax.set_title("")
            
            # 4. Tick Cleanup
            ax.tick_params(left=False) # Remove tick marks on Y-axis
            
            # Only show X-axis labels on the very bottom plot
            if i < 3:
                ax.set_xlabel('')
                ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            else:
                ax.set_xticklabels(layer_order, rotation=45, ha='right', fontsize=4)

        # 5. Global Title (Run Name)
        main_title = run_name if run_name else "Model Comparison"
        fig.suptitle(main_title, fontsize=16, y=0.92, fontweight='bold')
        
        # Colorbar Label
        cbar_ax.set_ylabel('Similarity Score (0.0 = Identical)', fontsize=10)

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path

    @staticmethod
    def _normalize_rows_to_unit_interval(matrix: np.ndarray) -> np.ndarray:
        m = matrix.copy()
        for i in range(m.shape[0]):
            row = m[i]
            mask = np.isfinite(row)
            if mask.sum() == 0:
                continue
            rmin = np.nanmin(row[mask])
            rmax = np.nanmax(row[mask])
            if rmax - rmin < 1e-12:
                m[i, mask] = 0.0
            else:
                m[i, mask] = (row[mask] - rmin) / (rmax - rmin)
        return m

    @staticmethod
    def plot_consolidated_row_normalized(
        results: Dict[str, Dict[str, Dict[str, float]]],
        comparisons: List[Tuple[str, str, str]],
        layer_order: List[str],
        output_path: str,
        run_name: str = None
    ):
        metric_order = [
            'Effective Scale/Bias',
            'Bhattacharyya Coeff',
            'W2 BN Stats',
            'CKA'
        ]

        fig, axes = plt.subplots(
            nrows=4,
            ncols=1,
            figsize=(24, 10),
            sharex=True,
            gridspec_kw={'hspace': 0}
        )

        cbar_ax = fig.add_axes([.91, .3, .015, .4])

        for i, metric_name in enumerate(metric_order):
            ax = axes[i]

            matrix = ComparisonVisualizer._build_matrix(
                results, metric_name, comparisons, layer_order
            )

            matrix = ComparisonVisualizer._normalize_rows_to_unit_interval(matrix)

            cmap = ComparisonVisualizer.COLORMAPS.get(metric_name, 'RdYlGn')
            base_cmap = plt.get_cmap(cmap).copy()
            base_cmap.set_bad(color='lightgrey')
            mask = np.isnan(matrix)

            sns.heatmap(
                matrix,
                xticklabels=layer_order,
                yticklabels=[],
                annot=False,
                cmap=base_cmap,
                vmin=0.0, vmax=1.0,
                ax=ax,
                mask=mask,
                cbar=(i == 0),
                cbar_ax=cbar_ax if i == 0 else None
            )

            ax.set_facecolor('lightgrey')
            ax.set_ylabel(metric_name, fontsize=11, fontweight='bold', labelpad=10)
            ax.set_title("")
            ax.tick_params(left=False)

            if i < 3:
                ax.set_xlabel('')
                ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            else:
                ax.set_xticklabels(layer_order, rotation=45, ha='right', fontsize=4)

        main_title = (run_name + " (Row-normalized)") if run_name else "Model Comparison (Row-normalized)"
        fig.suptitle(main_title, fontsize=16, y=0.92, fontweight='bold')
        cbar_ax.set_ylabel('Row-normalized (min→0, max→1)', fontsize=10)

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path



def create_comparison_report(
    results: Dict[str, Dict[str, Dict[str, float]]],
    comparisons: List[Tuple[str, str, str]],
    metric_names: List[str],
    num_samples: int,
    output_dir: str = "."
) -> str:
    """
    Create a text report of comparison results.
    
    Args:
        results: Comparison results
        comparisons: List of comparisons
        metric_names: List of metrics
        num_samples: Number of samples used
        output_dir: Directory for output
        
    Returns:
        Path to report file
    """
    import os
    
    report_path = os.path.join(output_dir, "comparison_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Samples processed: {num_samples}\n")
        f.write(f"Comparisons: {len(comparisons)}\n")
        f.write(f"Metrics: {', '.join(metric_names)}\n\n")
        
        # Write results per comparison
        for comp_title, model1, model2 in comparisons:
            f.write(f"\n--- Comparison: {comp_title} ---\n")
            f.write(f"  Model A: {model1}\n")
            f.write(f"  Model B: {model2}\n\n")
            
            if comp_title in results:
                for metric_name in metric_names:
                    if metric_name in results[comp_title]:
                        scores = results[comp_title][metric_name]
                        # Filter out NaN values for statistics
                        valid_scores = [v for v in scores.values() if not np.isnan(v)]
                        
                        f.write(f"  {metric_name}:\n")
                        f.write(f"    Layers: {len(scores)}\n")
                        f.write(f"    Valid:  {len(valid_scores)}\n")
                        
                        if len(valid_scores) > 0:
                            f.write(f"    Mean: {np.mean(valid_scores):.4f}\n")
                            f.write(f"    Std:  {np.std(valid_scores):.4f}\n")
                            f.write(f"    Min:  {np.min(valid_scores):.4f}\n")
                            f.write(f"    Max:  {np.max(valid_scores):.4f}\n\n")
                        else:
                            f.write(f"    No valid scores (all NaN)\n\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    return report_path