#!/usr/bin/env python3
"""
Main script for building multilayer networks from single-cell data.

This script processes single-cell RNA-seq data and constructs multilayer networks
based on ligand-receptor interactions between specified cell types.

Optimized for SLURM environments with proper figure saving and logging.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional, Set, List, Tuple, Dict, Any
import warnings
import os

import anndata
import numpy as np
import torch
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize
import json

# Configure matplotlib for headless environments
matplotlib.use('Agg')
plt.ioff()  # Turn off interactive mode

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parents[1]))

from mlnetst.core.knowledge.networks import load_resource
from mlnetst.core.network.build_network import assemble_multilayer_network
from mlnetst.utils.computation_utils import compute_tensor_memory_usage
from mlnetst.utils.mlnet_logging import get_colored_logger
from mlnetst.utils.build_network_utils import create_layer_gene_mapping
from mlnetst.utils.mlnet_utils import build_supra_adjacency_matrix_from_tensor
from mlnetst.utils.mlnet_metrics_utils import (
    compute_indegree,
    compute_instrength,
    compute_outdegree,
    compute_outstrength,
    compute_multi_indegree,
    compute_multi_outdegree,
    compute_multi_instrength,
    compute_multi_outstrength,
    compute_average_global_clustering
)

# Constants
RANDOM_STATE = 42
DEFAULT_DATA_PATH = Path(__file__).parents[1] / "data" / "processed" / "mouse1_slice153_x_hat_s.h5ad"
PROJECT_ROOT = Path(__file__).parents[1]
MEDIA_DIR = PROJECT_ROOT / "media"


class NetworkAnalyzer:
    """
    Enhanced network analyzer with parameterizable plotting options.
    
    This class provides flexible visualization methods for multilayer network analysis
    with customizable plot parameters, labels, colors, and styling options.
    """
    
    def __init__(self, 
                 experiment_name: str, 
                 logger: logging.Logger,
                 media_dir: Optional[Path] = None,
                 default_figsize: Tuple[int, int] = (12, 8),
                 default_dpi: int = 300,
                 default_style: str = "whitegrid"):
        """
        Initialize the NetworkAnalyzer with customizable defaults.
        
        Args:
            experiment_name: Name for the experiment (used in file naming)
            logger: Logger instance for output messages
            media_dir: Directory to save figures (if None, uses default)
            default_figsize: Default figure size as (width, height)
            default_dpi: Default DPI for saved figures
            default_style: Default seaborn style
        """
        self.experiment_name = experiment_name
        self.logger = logger
        self.default_figsize = default_figsize
        self.default_dpi = default_dpi
        
        # Set up media directory
        if media_dir is None:
            project_root = Path(__file__).parents[1]
            self.media_dir = project_root / "media"
        else:
            self.media_dir = Path(media_dir)
        
        self.media_dir.mkdir(exist_ok=True)
        
        # Set seaborn style
        sns.set_style(default_style)
        
    def _save_figure(self, 
                    fig_name: str, 
                    dpi: Optional[int] = None,
                    bbox_inches: str = 'tight',
                    format: str = 'png') -> Path:
        """
        Save figure with consistent naming and parameters.
        
        Args:
            fig_name: Base name for the figure
            dpi: DPI for saved figure (uses default if None)
            bbox_inches: Bounding box specification
            format: File format for saving
            
        Returns:
            Path to saved figure
        """
        dpi = dpi or self.default_dpi
        filename = f"{self.experiment_name}_{fig_name}.{format}"
        filepath = self.media_dir / filename
        
        plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, format=format)
        plt.close()
        self.logger.info(f"Figure saved: {filepath}")
        return filepath
    
    def plot_degree_distributions(self, 
                                  metrics_data: Dict[str, torch.Tensor],
                                  plot_config: Optional[Dict[str, Any]] = None) -> List[Path]:
        """
        Create customizable degree distribution plots.
        
        Args:
            metrics_data: Dictionary mapping metric names to tensor values
            plot_config: Configuration dictionary for plot customization
            
        Returns:
            List of paths to saved figures
        """
        # Default configuration
        default_config = {
            'figsize': self.default_figsize,
            'violin_colors': None,  # Use seaborn defaults
            'hist_colors': ['skyblue', 'lightcoral', 'lightgreen', 'gold'],
            'alpha': 0.7,
            'bins': 30,
            'edge_color': 'black',
            'title_fontsize': 14,
            'label_fontsize': 12,
            'violin_title': "Distribution of Network Metrics",
            'hist_title_template': "{metric} Distribution",
            'xlabel': "Metric Type",
            'ylabel': "Value",
            'hist_xlabel_template': "{metric}",
            'hist_ylabel': "Frequency"
        }
        
        # Update with user config
        config = {**default_config, **(plot_config or {})}
        
        self.logger.info("Creating degree distribution plots")
        
        # Prepare data for violin plots
        values = []
        metric_names = []
        for name, tensor in metrics_data.items():
            values.extend(tensor.cpu().numpy())
            metric_names.extend([name] * len(tensor))
        
        df_metrics = pd.DataFrame({
            "value": values,
            "metric": metric_names
        })
        
        # Create violin plot
        plt.figure(figsize=config['figsize'])
        violin_plot = sns.violinplot(data=df_metrics, x="metric", y="value", 
                                   palette=config['violin_colors'])
        plt.title(config['violin_title'], fontsize=config['title_fontsize'])
        plt.ylabel(config['ylabel'], fontsize=config['label_fontsize'])
        plt.xlabel(config['xlabel'], fontsize=config['label_fontsize'])
        plt.xticks(rotation=45)
        violin_path = self._save_figure("degree_distributions_violin")
        
        # Create histogram comparison
        num_metrics = len(metrics_data)
        cols = min(3, num_metrics)  # Max 3 columns
        rows = (num_metrics + cols - 1) // cols  # Ceiling division
        
        fig, axes = plt.subplots(rows, cols, figsize=(config['figsize'][0] * cols / 2, 
                                                     config['figsize'][1] * rows / 2))
        if num_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
        
        colors = config['hist_colors']
        
        for idx, (metric_name, tensor) in enumerate(metrics_data.items()):
            if idx < len(axes):
                ax = axes[idx]
                color = colors[idx % len(colors)]
                
                ax.hist(tensor.cpu().numpy(), 
                       bins=config['bins'], 
                       alpha=config['alpha'], 
                       color=color, 
                       edgecolor=config['edge_color'])
                
                ax.set_title(config['hist_title_template'].format(metric=metric_name.title()),
                           fontsize=config['title_fontsize'])
                ax.set_xlabel(config['hist_xlabel_template'].format(metric=metric_name.title()),
                            fontsize=config['label_fontsize'])
                ax.set_ylabel(config['hist_ylabel'], fontsize=config['label_fontsize'])
        
        # Hide extra subplots
        for idx in range(len(metrics_data), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        hist_path = self._save_figure("degree_distributions_histogram")
        
        return [violin_path, hist_path]
    
    def plot_spatial_networks(self, 
                            subdata: anndata.AnnData,
                            cell_indexes: List[str],
                            metrics_data: Dict[str, torch.Tensor],
                            plot_config: Optional[Dict[str, Any]] = None,
                            figname: str = "spatial_analysis") -> List[Path]:
        """
        Create enhanced spatial network visualizations.
        
        Args:
            subdata: AnnData object with spatial information
            cell_indexes: List of cell identifiers
            metrics_data: Dictionary mapping metric names to tensor values
            plot_config: Configuration dictionary for plot customization
            figname: Base name for the output figure
            
        Returns:
            List of paths to saved figures
        """
        # Default configuration
        default_config = {
            'figsize': (20, 16),
            'scatter_size': 25,  # Reduced default size
            'scatter_alpha': 0.6,
            'correlation_scatter_size': 20,  # Even smaller for correlation plots
            'correlation_alpha': 0.5,
            'colormaps': ['viridis', 'plasma', 'cividis', 'magma'],
            'coord_columns': ['centroid_x', 'centroid_y'],
            'xlabel': "X Coordinate",
            'ylabel': "Y Coordinate",
            'title_fontsize': 14,
            'label_fontsize': 12,
            'colorbar_label_fontsize': 10,
            'trend_line_color': 'red',
            'trend_line_style': '--',
            'trend_line_alpha': 0.8,
            'correlation_precision': 3,
            'subplot_titles': None,  # Auto-generate if None
            'show_correlation_matrix': True,
            'combined_view_markers': ['o', 's', '^', 'D'],
            'combined_view_sizes': [60, 40, 45, 35]
        }
        
        # Update with user config
        config = {**default_config, **(plot_config or {})}
        
        self.logger.info("Creating spatial network plots")
        
        # Create graph with spatial positions and metrics
        g = nx.Graph()
        coord_x, coord_y = config['coord_columns']
        
        nodes_dict = {}
        for idx, cell_id in enumerate(cell_indexes):
            node_data = {
                "x": subdata.obs.loc[cell_id, coord_x],
                "y": subdata.obs.loc[cell_id, coord_y]
            }
            # Add all metrics to node data
            for metric_name, tensor in metrics_data.items():
                node_data[metric_name] = float(tensor[idx])
            nodes_dict[cell_id] = node_data
        
        g.add_nodes_from(nodes_dict.items())
        
        # Extract positions and metric values
        pos = {cell_id: (nodes_dict[cell_id]["x"], nodes_dict[cell_id]["y"]) 
               for cell_id in nodes_dict}
        
        metric_values = {}
        for metric_name in metrics_data.keys():
            metric_values[metric_name] = [nodes_dict[cell_id][metric_name] 
                                        for cell_id in nodes_dict]
        
        # Log statistics
        self._log_network_stats(metric_values)
        
        # Determine subplot layout
        num_metrics = len(metrics_data)
        if config['show_correlation_matrix'] and num_metrics > 1:
            # Include correlation plots
            total_plots = num_metrics + min(3, num_metrics * (num_metrics - 1) // 2)
        else:
            total_plots = num_metrics
        
        # Calculate optimal subplot layout
        cols = min(3, total_plots)
        rows = (total_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=config['figsize'])
        if total_plots == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
        
        plot_idx = 0
        colormaps = config['colormaps']
        
        # Individual metric spatial plots
        for idx, (metric_name, values) in enumerate(metric_values.items()):
            if plot_idx < len(axes):
                ax = axes[plot_idx]
                cmap = colormaps[idx % len(colormaps)]
                
                scatter = ax.scatter([pos[cell_id][0] for cell_id in nodes_dict], 
                                   [pos[cell_id][1] for cell_id in nodes_dict],
                                   c=values, 
                                   cmap=cmap, 
                                   s=config['scatter_size'], 
                                   alpha=config['scatter_alpha'])
                
                title = (config['subplot_titles'][idx] if config['subplot_titles'] 
                        else f"Spatial {metric_name.replace('_', ' ').title()} Distribution")
                ax.set_title(title, fontsize=config['title_fontsize'])
                ax.set_xlabel(config['xlabel'], fontsize=config['label_fontsize'])
                ax.set_ylabel(config['ylabel'], fontsize=config['label_fontsize'])
                
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(metric_name.replace('_', ' ').title(), 
                              fontsize=config['colorbar_label_fontsize'])
                
                plot_idx += 1
        
        # Correlation plots (if enabled and multiple metrics)
        if config['show_correlation_matrix'] and len(metric_values) > 1:
            metric_names = list(metric_values.keys())
            correlation_count = 0
            max_correlations = min(3, len(axes) - plot_idx)  # Limit to available space
            
            for i in range(len(metric_names)):
                for j in range(i + 1, len(metric_names)):
                    if correlation_count >= max_correlations or plot_idx >= len(axes):
                        break
                    
                    ax = axes[plot_idx]
                    x_vals = metric_values[metric_names[i]]
                    y_vals = metric_values[metric_names[j]]
                    
                    correlation = np.corrcoef(x_vals, y_vals)[0, 1]
                    
                    ax.scatter(x_vals, y_vals, 
                             alpha=config['correlation_alpha'], 
                             s=config['correlation_scatter_size'])
                    
                    # Add trend line
                    z = np.polyfit(x_vals, y_vals, 1)
                    p = np.poly1d(z)
                    ax.plot(x_vals, p(x_vals), 
                           color=config['trend_line_color'],
                           linestyle=config['trend_line_style'], 
                           alpha=config['trend_line_alpha'])
                    
                    ax.set_xlabel(metric_names[i].replace('_', ' ').title(),
                                fontsize=config['label_fontsize'])
                    ax.set_ylabel(metric_names[j].replace('_', ' ').title(),
                                fontsize=config['label_fontsize'])
                    
                    correlation_str = f"r={correlation:.{config['correlation_precision']}f}"
                    title = f"{metric_names[i].title()} vs {metric_names[j].title()} ({correlation_str})"
                    ax.set_title(title, fontsize=config['title_fontsize'])
                    
                    plot_idx += 1
                    correlation_count += 1
                
                if correlation_count >= max_correlations:
                    break
        
        # Hide extra subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        spatial_path = self._save_figure(f"spatial_network_analysis_{figname}")
        
        return [spatial_path]
    
    def plot_correlation_matrix(self,
                               metrics_data: Dict[str, torch.Tensor],
                               plot_config: Optional[Dict[str, Any]] = None) -> List[Path]:
        """
        Create a comprehensive correlation matrix heatmap.
        
        Args:
            metrics_data: Dictionary mapping metric names to tensor values
            plot_config: Configuration dictionary for plot customization
            
        Returns:
            List of paths to saved figures
        """
        default_config = {
            'figsize': (10, 8),
            'cmap': 'coolwarm',
            'center': 0,
            'annot': True,
            'fmt': '.3f',
            'title': 'Network Metrics Correlation Matrix',
            'title_fontsize': 16,
            'annot_fontsize': 10
        }
        
        config = {**default_config, **(plot_config or {})}
        
        # Create correlation matrix
        df_data = {}
        for name, tensor in metrics_data.items():
            df_data[name.replace('_', ' ').title()] = tensor.cpu().numpy()
        
        df = pd.DataFrame(df_data)
        correlation_matrix = df.corr()
        
        # Create heatmap
        plt.figure(figsize=config['figsize'])
        sns.heatmap(correlation_matrix, 
                   cmap=config['cmap'],
                   center=config['center'],
                   annot=config['annot'],
                   fmt=config['fmt'],
                   square=True,
                   annot_kws={'fontsize': config['annot_fontsize']})
        
        plt.title(config['title'], fontsize=config['title_fontsize'])
        plt.tight_layout()
        
        correlation_path = self._save_figure("correlation_matrix")
        return [correlation_path]
    
    def create_custom_plot(self,
                          plot_func: callable,
                          plot_args: tuple = (),
                          plot_kwargs: Optional[Dict[str, Any]] = None,
                          fig_name: str = "custom_plot",
                          figsize: Optional[Tuple[int, int]] = None) -> Path:
        """
        Create a custom plot using a user-defined function.
        
        Args:
            plot_func: Function that creates the plot
            plot_args: Positional arguments for plot_func
            plot_kwargs: Keyword arguments for plot_func
            fig_name: Name for the saved figure
            figsize: Figure size (uses default if None)
            
        Returns:
            Path to saved figure
        """
        figsize = figsize or self.default_figsize
        plot_kwargs = plot_kwargs or {}
        
        plt.figure(figsize=figsize)
        plot_func(*plot_args, **plot_kwargs)
        
        custom_path = self._save_figure(fig_name)
        return custom_path
    
    def _log_network_stats(self, metric_values: Dict[str, List[float]]):
        """
        Log comprehensive network statistics.
        
        Args:
            metric_values: Dictionary mapping metric names to value lists
        """
        self.logger.info("Network Statistics Summary:")
        self.logger.info("-" * 50)
        
        for metric_name, values in metric_values.items():
            values_array = np.array(values)
            stats = {
                'Min': np.min(values_array),
                'Max': np.max(values_array),
                'Mean': np.mean(values_array),
                'Median': np.median(values_array),
                'Std': np.std(values_array),
                'Q1': np.percentile(values_array, 25),
                'Q3': np.percentile(values_array, 75)
            }
            
            self.logger.info(f"{metric_name.upper()}:")
            for stat_name, stat_value in stats.items():
                self.logger.info(f"  {stat_name}: {stat_value:.3f}")
            self.logger.info("")


# Example usage function to replace the original analyze_and_visualize_network
def analyze_and_visualize_network_enhanced(
    mlnet: torch.Tensor,
    subdata: anndata.AnnData,
    cell_indexes: List[str],
    args: argparse.Namespace,
    logger: logging.Logger
) -> None:
    """
    Enhanced network analysis with improved NetworkAnalyzer.
    
    This replaces the original analyze_and_visualize_network function
    with more flexible plotting options.
    """
    logger.info("Starting enhanced network analysis and visualization")
    
    # Compute network metrics (same as before)
    logger.info("Computing network metrics")
    num_layers = mlnet.shape[1]
    num_cells = mlnet.shape[0]
    
    # Import required functions (you'll need to ensure these are available)
    from mlnetst.utils.mlnet_utils import build_supra_adjacency_matrix_from_tensor
    from mlnetst.utils.mlnet_metrics_utils import (
        compute_instrength, compute_multi_instrength,
        compute_outstrength, compute_multi_outstrength,
        compute_indegree, compute_multi_indegree,
        compute_outdegree, compute_multi_outdegree
    )
    
    supra_adjacency = build_supra_adjacency_matrix_from_tensor(mlnet)
    
    # Compute all metrics
    metrics_data = {
        'in_strength': compute_instrength(supra_adjacency, num_cells, num_layers),
        'out_strength': compute_outstrength(supra_adjacency, num_cells, num_layers),
        'in_degree': compute_indegree(supra_adjacency, num_cells, num_layers),
        'out_degree': compute_outdegree(supra_adjacency, num_cells, num_layers),
        'multi_in_strength': compute_multi_instrength(supra_adjacency, num_cells, num_layers),
        'multi_out_strength': compute_multi_outstrength(supra_adjacency, num_cells, num_layers),
        'multi_in_degree': compute_multi_indegree(supra_adjacency, num_cells, num_layers),
        'multi_out_degree': compute_multi_outdegree(supra_adjacency, num_cells, num_layers),
    }
    
    logger.info("✅ Network metrics computed")
    
    # Create enhanced analyzer
    analyzer = NetworkAnalyzer(
        experiment_name=args.experiment_name,
        logger=logger,
        default_figsize=(14, 10),  # Larger default figures
        default_dpi=300
    )
    
    # Custom plot configurations
    degree_config = {
        'figsize': (15, 8),
        'bins': 40,
        'alpha': 0.75,
        'violin_title': "Distribution of Multilayer Network Metrics",
        'xlabel': "Network Metric",
        'ylabel': "Metric Value",
        'title_fontsize': 16,
        'label_fontsize': 14
    }
    
    spatial_config = {
        'figsize': (24, 18),
        'scatter_size': 20,  # Smaller dots as requested
        'scatter_alpha': 0.65,
        'correlation_scatter_size': 15,  # Even smaller for correlations
        'xlabel': "Spatial X Coordinate (μm)",
        'ylabel': "Spatial Y Coordinate (μm)",
        'title_fontsize': 14,
        'label_fontsize': 12,
        'show_correlation_matrix': True
    }
    
    # Generate enhanced plots
    logger.info("Creating degree distribution plots")
    dist_plots = analyzer.plot_degree_distributions(metrics_data, degree_config)
    
    logger.info("Creating spatial network visualizations")
    
    # Create separate spatial plots for different metric groups
    basic_metrics = {k: v for k, v in metrics_data.items() 
                    if not k.startswith('multi_')}
    multi_metrics = {k: v for k, v in metrics_data.items() 
                    if k.startswith('multi_')}
    
    spatial_plots = []
    spatial_plots.extend(analyzer.plot_spatial_networks(
        subdata, cell_indexes, basic_metrics, spatial_config, "basic_metrics"
    ))
    spatial_plots.extend(analyzer.plot_spatial_networks(
        subdata, cell_indexes, multi_metrics, spatial_config, "multi_metrics"
    ))
    
    # Create correlation matrix
    logger.info("Creating correlation matrix")
    correlation_plots = analyzer.plot_correlation_matrix(
        metrics_data, 
        {'figsize': (12, 10), 'title_fontsize': 16}
    )
    
    logger.info(f"✅ Created {len(dist_plots)} distribution plots, "
               f"{len(spatial_plots)} spatial plots, and "
               f"{len(correlation_plots)} correlation plots")
    logger.info("✅ Enhanced network analysis and visualization completed")


def setup_logging(verbose_level: int, experiment_name: str) -> Tuple[logging.Logger, logging.Logger, logging.Logger, str]:
    """Set up structured logging with separate loggers for different components."""
    log_mode = "debug" if verbose_level >= 1 else "info"
    log_level = logging.DEBUG if verbose_level >= 1 else logging.INFO
    
    # Create separate loggers for different components
    main_logger = get_colored_logger("MAIN", level=log_level, mode=log_mode)
    mapping_logger = get_colored_logger("LAYER_MAPPING", level=log_level, mode=log_mode)
    network_logger = get_colored_logger("NETWORK", level=log_level, mode=log_mode)
    
    # Log experiment info
    main_logger.info(f"Starting experiment: {experiment_name}")
    main_logger.info(f"Log mode: {log_mode}")
    
    return main_logger, mapping_logger, network_logger, log_mode


def str_or_none(value: str) -> Optional[str]:
    """Convert string to None if it equals 'None' (case-insensitive)."""
    return value if value.lower() != "none" else None


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments with enhanced validation."""
    parser = argparse.ArgumentParser(
        description="Build multilayer networks from single-cell data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--num_layers", "-L", 
        type=int, 
        required=True,
        help="Number of layers in the multilayer network"
    )
    
    parser.add_argument(
        "--num_cells", "-N", 
        type=int, 
        required=True,
        help="Number of cells to sample for analysis"
    )
    
    # Optional arguments with better defaults
    parser.add_argument(
        "--mode", "-m", 
        type=str, 
        default="tensor", 
        choices=["tensor", "pymnet_multinetwork"],
        help="Mode to build the network"
    )
    
    parser.add_argument(
        "--resource_lr", "-lr", 
        type=str, 
        default="mouseconsensus",
        choices=["nichenet", "mouseconsensus"],
        help="Resource for ligand-receptor interactions"
    )
    
    parser.add_argument(
        "--resource_grn", "-grn", 
        type=str, 
        default="nichenet",
        choices=["nichenet", "mouseconsensus", "scseqcomm"],
        help="Resource for gene regulatory network interactions"
    )
    
    parser.add_argument(
        "--verbose", "-v", 
        type=int, 
        default=0, 
        help="Verbosity level: 0=INFO, 1+=DEBUG"
    )
    
    parser.add_argument(
        "--source_cell_type", 
        type=str_or_none,
        default=None,
        help="Source cell type (use 'None' for all types)"
    )

    parser.add_argument(
        "--target_cell_type", 
        type=str_or_none,
        default=None,
        help="Target cell type (use 'None' for all types)"
    )
    
    parser.add_argument(
        "--inter_coupling", 
        type=str, 
        default="rtl", 
        choices=["rtl", "combinatorial"],
        help="Inter-layer coupling type"
    )
    
    parser.add_argument(
        "--data_path", 
        type=Path, 
        default=DEFAULT_DATA_PATH,
        help="Path to input AnnData file"
    )
    
    parser.add_argument(
        "--force", 
        action="store_true", 
        default=False,
        help="Force overwrite existing files"
    )
    
    parser.add_argument(
        "--force_all",
        action="store_true",
        default=False,
        help="Force overwrite all existing files"
    )
    
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="mlnet_experiment",
        help="Name for this experiment (used in output files)"
    )
    
    parser.add_argument(
        "--min_gene_expression",
        type=float,
        default=25.0,
        help="Minimum gene expression threshold"
    )
    
    parser.add_argument(
        "--min_genes_per_cell",
        type=int,
        default=100,
        help="Minimum genes per cell for filtering"
    )
    
    parser.add_argument(
        "--min_cells_per_gene",
        type=int,
        default=3,
        help="Minimum cells per gene for filtering"
    )
    parser.add_argument(
        "--th_sparsify_weight_resource",
        type=float,
        default=0.05,
        help="Threshold for sparsifying weight interactions"
    )
    parser.add_argument(
        "--th_sparsify_degree_resource",
        type=float,
        default=0.05,
        help="Threshold for sparsifying degree interactions"
    )
    
    parser.add_argument(
        "--radius",
        type=float,
        default=torch.inf,
        help="Radius for distance computation"
    )
    
    parser.add_argument(
        "--gr_extra_params",
        type=str,
        default="all",
        help="Additional parameters for resource loading (e.g., 'all' for all parameters)"
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Validate command line arguments."""
    if args.num_layers <= 0:
        raise ValueError("Number of layers must be positive")
    if args.num_cells <= 0:
        raise ValueError("Number of cells must be positive")
    if not args.data_path.exists():
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    
    logger.info("✅ Arguments validated successfully")


def load_and_filter_data(
    data_path: Path, 
    source_type: Optional[str], 
    target_type: Optional[str],
    logger: logging.Logger
) -> anndata.AnnData:
    """Load and filter single-cell data."""
    logger.info(f"Loading data from: {data_path}")
    
    try:
        x_hat_s = anndata.read_h5ad(data_path)
        logger.info(f"✅ Loaded data with shape: {x_hat_s.shape}")
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        raise
    
    # Filter for specified cell types
    if source_type is None or target_type is None:
        cell_types = x_hat_s.obs["subclass"].unique().tolist()
        logger.info(f"Using all available cell types: {cell_types}")
        subdata = x_hat_s
    else:
        cell_types = [source_type, target_type]
        subdata = x_hat_s[x_hat_s.obs["subclass"].isin(cell_types), :]
        logger.info(f"Filtered for cell types: {cell_types}")
    
    logger.info(f"Final filtered data shape: {subdata.shape}")
    logger.debug(f"Cell type distribution:\n{subdata.obs['subclass'].value_counts()}")
    
    return subdata


def perform_quality_control(
    subdata: anndata.AnnData, 
    args: argparse.Namespace,
    logger: logging.Logger
) -> anndata.AnnData:
    """Perform quality control on the data."""
    logger.info("Starting quality control analysis")
    
    # Apply expression threshold
    original_shape = subdata.shape
    subdata.X[subdata.X < args.min_gene_expression] = 0
    logger.info(f"Applied expression threshold: {args.min_gene_expression}")
    
    # Calculate QC metrics
    sc.pp.calculate_qc_metrics(subdata, inplace=True)
    logger.info("Calculated initial QC metrics")
    
    # Apply filters
    sc.pp.filter_cells(subdata, min_genes=args.min_genes_per_cell)
    sc.pp.filter_genes(subdata, min_cells=args.min_cells_per_gene)
    
    logger.info(f"QC filtering complete: {original_shape} -> {subdata.shape}")
    logger.info(f"Removed {original_shape[0] - subdata.shape[0]} cells and "
               f"{original_shape[1] - subdata.shape[1]} genes")
    
    # Recalculate QC metrics after filtering
    sc.pp.calculate_qc_metrics(subdata, inplace=True)
    
    return subdata


def sample_cells(
    subdata: anndata.AnnData, 
    num_cells: int,
    logger: logging.Logger
) -> List[str]:
    """Sample cells from the dataset with proper handling."""
    total_cells = len(subdata.obs_names)
    
    if num_cells >= total_cells:
        logger.warning(f"Requested {num_cells} cells but only {total_cells} available. Using all cells.")
        return subdata.obs_names.tolist()
    
    logger.info(f"Sampling {num_cells} cells from {total_cells} available")
    sampled_cells = subdata.obs.sample(
        num_cells, replace=False, random_state=RANDOM_STATE
    ).index.tolist()
    
    return sampled_cells


def is_valid_complex(gene_string: str, valid_genes: Set[str]) -> bool:
    """Check if gene complex components are valid."""
    components = gene_string.split("_")
    return all(comp.lower() in valid_genes for comp in components)


def filter_lr_interactions(
    subdata: anndata.AnnData, 
    num_layers: int,
    resource: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """Load and filter ligand-receptor interactions."""
    logger.info(f"Loading {resource} ligand-receptor database")
    
    try:
        lr_interactions_df = load_resource(resource)
        logger.info(f"✅ Loaded {len(lr_interactions_df)} total interactions")
    except Exception as e:
        logger.error(f"❌ Failed to load resource {resource}: {e}")
        raise
    
    # Create set of valid genes
    valid_genes = set(g.lower() for g in subdata.var_names)
    logger.debug(f"Found {len(valid_genes)} valid genes in dataset")
    
    # Filter interactions
    logger.info("Filtering interactions for valid gene complexes")
    filtered_df = lr_interactions_df[
        lr_interactions_df["source"].apply(lambda x: is_valid_complex(x, valid_genes)) &
        lr_interactions_df["target"].apply(lambda x: is_valid_complex(x, valid_genes))
    ]
    
    logger.info(f"Found {len(filtered_df)} valid interactions")
    
    if len(filtered_df) < num_layers:
        logger.warning(f"Only {len(filtered_df)} interactions available for {num_layers} requested layers")
        return filtered_df
    
    # Sample required interactions
    sample_lr = filtered_df.sample(n=num_layers, random_state=RANDOM_STATE).reset_index(drop=True)
    logger.info(f"✅ Sampled {len(sample_lr)} interactions for network layers")
    
    return sample_lr


def build_multilayer_network(
    subdata: anndata.AnnData,
    cell_indexes: List[str],
    lr_interactions: pd.DataFrame,
    args: argparse.Namespace,
    mapping_logger: logging.Logger,
    network_logger: logging.Logger
) -> torch.Tensor:
    """Build the multilayer network."""
    network_logger.info("Starting multilayer network construction")
    
    # Create layer mapping
    mapping_logger.info("Creating layer-gene mapping")
    if (PROJECT_ROOT / "data" / "processed" / f"{args.experiment_name}_layer_mapping.csv").exists() and not args.force_all:
        layer_mapping = json.load(open(PROJECT_ROOT / "data" / "processed" / f"{args.experiment_name}_layer_mapping.json"))
        mapping_logger.info(f"Loaded existing layer mapping with {len(layer_mapping)} layers")
    else:
        mapping_logger.warning(f"Creating new layer mapping..., this may take a while")
        layer_mapping = create_layer_gene_mapping(
            ligand_ids=lr_interactions["source"].str.lower().unique().tolist(),
            receptor_ids=lr_interactions["target"].str.lower().unique().tolist(),
            var_names=subdata.var_names,
            resource=args.resource_grn,
            resource_extra_params=args.gr_extra_params,
            inter_coupling=args.inter_coupling,
            th_sparsify_weight_resource=args.th_sparsify_weight_resource,
            th_sparsify_degree_resource=args.th_sparsify_degree_resource,
            logger=mapping_logger
        )
        mapping_logger.info(f"✅ Created mapping for {len(layer_mapping)} layers")
        # Save layer mapping
        mapping_path = PROJECT_ROOT / "data" / "processed" / f"{args.experiment_name}_layer_mapping.json"
        with open(mapping_path, "w") as f:
            json.dump(layer_mapping, f)
        mapping_logger.info(f"Layer mapping saved to: {mapping_path}")
    
    # Build network
    network_logger.debug(f"Layer mapping: {layer_mapping}")
    network_logger.info("Assembling multilayer network")
    mlnet = assemble_multilayer_network(
        subdata[cell_indexes, :], 
        layer_mapping,
        toll_complexes=1e-6,
        toll_distance=1e-6,
        radius=args.radius,
        build_intra=True,
        build_inter=True,
        logger=network_logger,
        mode=args.mode,
    )
    
    network_logger.info("✅ Multilayer network construction completed")
    

    
    return mlnet

def save_network_data(
    mlnet: torch.Tensor,
    subdata: anndata.AnnData,
    experiment_name: str,
    logger: logging.Logger
) -> Path:
    """Save network data to disk."""
    network_path = PROJECT_ROOT / "data" / "processed" / f"{experiment_name}_mlnet.pth"
    torch.save(mlnet, network_path)
    subdata.write_h5ad(PROJECT_ROOT / "data" / "processed" / f"{experiment_name}_subdata.h5ad")
    logger.info(f"✅ Multilayer network saved to: {network_path}")
    return network_path


def main() -> None:
    """Main execution function."""
    try:
        # Parse and validate arguments
        args = parse_arguments()
        
        # Set up logging
        main_logger, mapping_logger, network_logger, _ = setup_logging(
            args.verbose, args.experiment_name
        )
        
        validate_arguments(args, main_logger)
        
        # Set random seeds
        main_logger.info(f"Setting random seed to {RANDOM_STATE}")
        np.random.seed(RANDOM_STATE)
        torch.manual_seed(RANDOM_STATE)
        
        main_logger.info(f"Parameters: N={args.num_cells}, L={args.num_layers}, "
                        f"mode={args.mode}, resource_lr={args.resource_lr}, resource_grn={args.resource_grn}")
        
        # Load and process data
        subdata = load_and_filter_data(
            args.data_path, 
            args.source_cell_type, 
            args.target_cell_type,
            main_logger
        )
        
        # Quality control
        subdata = perform_quality_control(subdata, args, main_logger)
        
        # Filter ligand-receptor interactions
        lr_interactions = filter_lr_interactions(
            subdata, args.num_layers, args.resource_lr, main_logger
        )
        
        main_logger.info(f"Filtered ligand-receptor interactions: {len(lr_interactions)} valid interactions")
        
        # Sample cells
        cell_indexes = sample_cells(subdata, args.num_cells, main_logger)
        final_num_cells = len(cell_indexes)
        main_logger.info(f"Using {final_num_cells} cells for analysis")
        
        # Check if network already exists
        network_path = PROJECT_ROOT / "data" / "processed" / f"{args.experiment_name}_mlnet.pth"
        
        if network_path.exists() and not args.force:
            main_logger.info("Loading existing multilayer network (use --force to rebuild)")
            mlnet = torch.load(network_path)
        else:
            # Build new network
            mlnet = build_multilayer_network(
                subdata, cell_indexes, lr_interactions, args,
                mapping_logger, network_logger
            )
            # Save network
            save_network_data(mlnet, subdata, args.experiment_name, main_logger)
        
        # Analyze and visualize
        # Update args.num_cells to reflect actual number of cells used
        args.num_cells = final_num_cells
        analyze_and_visualize_network_enhanced(
            mlnet, subdata, cell_indexes, args, main_logger
        )
        
        main_logger.info("🎉 Analysis completed successfully!")
        
    except Exception as e:
        if 'main_logger' in locals():
            main_logger.error(f"❌ Fatal error: {str(e)}")
            main_logger.debug("Full traceback:", exc_info=True)
        else:
            print(f"❌ Fatal error during setup: {str(e)}")
            
        sys.exit(1)


if __name__ == "__main__":
    main()