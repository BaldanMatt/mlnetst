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
    compute_multi_indegree,
    compute_multi_outdegree,
    compute_total_degree,
    compute_average_global_clustering
)

# Constants
RANDOM_STATE = 42
DEFAULT_DATA_PATH = Path(__file__).parents[1] / "data" / "processed" / "mouse1_slice153_x_hat_s.h5ad"
PROJECT_ROOT = Path(__file__).parents[1]
MEDIA_DIR = PROJECT_ROOT / "media"


class NetworkAnalyzer:
    """Handle network analysis and visualization."""
    
    def __init__(self, experiment_name: str, logger: logging.Logger):
        self.experiment_name = experiment_name
        self.logger = logger
        self.media_dir = MEDIA_DIR
        self.media_dir.mkdir(exist_ok=True)
        
    def _save_figure(self, fig_name: str, dpi: int = 300) -> Path:
        """Save figure with consistent naming."""
        filename = f"{self.experiment_name}_{fig_name}.png"
        filepath = self.media_dir / filename
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Figure saved: {filepath}")
        return filepath
        
    def plot_degree_distributions(self, 
                                  in_strength: torch.Tensor, 
                                  in_degree: torch.Tensor) -> List[Path]:
        """Create and save degree distribution plots."""
        self.logger.info("Creating degree distribution plots")
        
        # Prepare data for violin plots
        df_metrics = pd.DataFrame({
            "value": np.concatenate([in_strength.cpu().numpy(), in_degree.cpu().numpy()]),
            "metric": ["instrength"] * len(in_strength) + ["indegree"] * len(in_degree)
        })
        
        # Create violin plot
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=df_metrics, x="metric", y="value")
        plt.title("Distribution of Network Metrics")
        plt.ylabel("Value")
        plt.xlabel("Metric Type")
        violin_path = self._save_figure("degree_distributions_violin")
        
        # Create histogram comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        axes[0].hist(in_strength.cpu().numpy(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_title("In-Strength Distribution")
        axes[0].set_xlabel("In-Strength")
        axes[0].set_ylabel("Frequency")
        
        axes[1].hist(in_degree.cpu().numpy(), bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1].set_title("In-Degree Distribution")
        axes[1].set_xlabel("In-Degree")
        axes[1].set_ylabel("Frequency")
        
        plt.tight_layout()
        hist_path = self._save_figure("degree_distributions_histogram")
        
        return [violin_path, hist_path]
        
    def plot_spatial_networks(self, 
                            subdata: anndata.AnnData,
                            cell_indexes: List[str],
                            in_strength: torch.Tensor,
                            in_degree: torch.Tensor) -> List[Path]:
        """Create spatial network visualizations."""
        self.logger.info("Creating spatial network plots")
        
        # Create graph with spatial positions
        g = nx.Graph()
        nodes_dict = {
            cell_id: {
                "x": subdata.obs.loc[cell_id, "centroid_x"],
                "y": subdata.obs.loc[cell_id, "centroid_y"],
                "instrength": float(in_strength[idx]),
                "indegree": float(in_degree[idx])
            } for idx, cell_id in enumerate(cell_indexes)
        }
        g.add_nodes_from(nodes_dict.items())
        
        # Extract values and statistics
        pos = {cell_id: (nodes_dict[cell_id]["x"], nodes_dict[cell_id]["y"]) 
               for cell_id in nodes_dict}
        instrength_values = [nodes_dict[cell_id]["instrength"] for cell_id in nodes_dict]
        indegree_values = [nodes_dict[cell_id]["indegree"] for cell_id in nodes_dict]
        
        # Log statistics
        self._log_network_stats(instrength_values, indegree_values)
        
        # Create comprehensive spatial visualization
        fig, axs = plt.subplots(2, 2, figsize=(20, 16))
        
        # Plot 1: In-strength spatial distribution
        scatter1 = axs[0, 0].scatter([pos[i][0] for i in nodes_dict], 
                                   [pos[i][1] for i in nodes_dict],
                                   c=instrength_values, cmap='viridis', 
                                   s=50, alpha=0.7)
        axs[0, 0].set_title("Spatial In-Strength Distribution")
        axs[0, 0].set_xlabel("X Coordinate")
        axs[0, 0].set_ylabel("Y Coordinate")
        plt.colorbar(scatter1, ax=axs[0, 0], label='In-Strength')
        
        # Plot 2: In-degree spatial distribution
        scatter2 = axs[0, 1].scatter([pos[i][0] for i in nodes_dict], 
                                   [pos[i][1] for i in nodes_dict],
                                   c=indegree_values, cmap='plasma', 
                                   s=50, alpha=0.7)
        axs[0, 1].set_title("Spatial In-Degree Distribution")
        axs[0, 1].set_xlabel("X Coordinate")
        axs[0, 1].set_ylabel("Y Coordinate")
        plt.colorbar(scatter2, ax=axs[0, 1], label='In-Degree')
        
        # Plot 3: Correlation scatter plot
        correlation = np.corrcoef(instrength_values, indegree_values)[0, 1]
        axs[1, 0].scatter(instrength_values, indegree_values, alpha=0.6, s=30)
        axs[1, 0].set_xlabel('In-Strength')
        axs[1, 0].set_ylabel('In-Degree')
        axs[1, 0].set_title(f'In-Strength vs In-Degree (r={correlation:.3f})')
        
        # Add trend line
        z = np.polyfit(instrength_values, indegree_values, 1)
        p = np.poly1d(z)
        axs[1, 0].plot(instrength_values, p(instrength_values), "r--", alpha=0.8)
        
        # Plot 4: Combined spatial view with different markers
        axs[1, 1].scatter([pos[i][0] for i in nodes_dict], 
                         [pos[i][1] for i in nodes_dict],
                         c=instrength_values, cmap='viridis', 
                         s=80, alpha=0.6, marker='o', label='In-Strength')
        axs[1, 1].scatter([pos[i][0] for i in nodes_dict], 
                         [pos[i][1] for i in nodes_dict],
                         c=indegree_values, cmap='plasma', 
                         s=40, alpha=0.8, marker='s', label='In-Degree')
        axs[1, 1].set_title("Combined Spatial Distribution")
        axs[1, 1].set_xlabel("X Coordinate")
        axs[1, 1].set_ylabel("Y Coordinate")
        axs[1, 1].legend()
        
        plt.tight_layout()
        spatial_path = self._save_figure("spatial_network_analysis")
        
        return [spatial_path]
        
    def _log_network_stats(self, instrength_values: List[float], indegree_values: List[float]):
        """Log network statistics."""
        self.logger.info("Network Statistics:")
        self.logger.info(f"  In-strength - Min: {min(instrength_values):.3f}, "
                        f"Max: {max(instrength_values):.3f}, "
                        f"Mean: {np.mean(instrength_values):.3f}, "
                        f"Std: {np.std(instrength_values):.3f}")
        self.logger.info(f"  In-degree - Min: {min(indegree_values):.3f}, "
                        f"Max: {max(indegree_values):.3f}, "
                        f"Mean: {np.mean(indegree_values):.3f}, "
                        f"Std: {np.std(indegree_values):.3f}")


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
        choices=["nichenet", "mouseconsensus"],
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
            inter_coupling=args.inter_coupling,
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
    input("Waiting for user input before building network...")
    network_logger.info("Assembling multilayer network")
    mlnet = assemble_multilayer_network(
        subdata[cell_indexes, :], 
        layer_mapping,
        toll_complexes=1e-6,
        toll_distance=1e-6,
        build_intra=True,
        build_inter=True,
        logger=network_logger,
        mode=args.mode,
    )
    
    network_logger.info("✅ Multilayer network construction completed")
    

    
    return mlnet


def analyze_and_visualize_network(
    mlnet: torch.Tensor,
    subdata: anndata.AnnData,
    cell_indexes: List[str],
    args: argparse.Namespace,
    logger: logging.Logger
) -> None:
    """Analyze network and create visualizations."""
    logger.info("Starting network analysis and visualization")
    
    # Compute network metrics
    logger.info("Computing network metrics")
    supra_adjacency = build_supra_adjacency_matrix_from_tensor(mlnet)
    in_strength = compute_instrength(supra_adjacency, args.num_cells, args.num_layers)
    in_degree = compute_indegree(supra_adjacency, args.num_cells, args.num_layers)
    
    logger.info("✅ Network metrics computed")
    
    # Create analyzer and generate plots
    analyzer = NetworkAnalyzer(args.experiment_name, logger)
    
    # Generate degree distribution plots
    dist_plots = analyzer.plot_degree_distributions(in_strength, in_degree)
    logger.info(f"Created {len(dist_plots)} degree distribution plots")
    
    # Generate spatial network plots
    spatial_plots = analyzer.plot_spatial_networks(subdata, cell_indexes, in_strength, in_degree)
    logger.info(f"Created {len(spatial_plots)} spatial network plots")
    
    logger.info("✅ Network analysis and visualization completed")


def save_network_data(
    mlnet: torch.Tensor,
    experiment_name: str,
    logger: logging.Logger
) -> Path:
    """Save network data to disk."""
    network_path = PROJECT_ROOT / "data" / "processed" / f"{experiment_name}_mlnet.pth"
    torch.save(mlnet, network_path)
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
        input("Waiting...")
        
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
            save_network_data(mlnet, args.experiment_name, main_logger)
        
        # Analyze and visualize
        # Update args.num_cells to reflect actual number of cells used
        args.num_cells = final_num_cells
        analyze_and_visualize_network(
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