#!/usr/bin/env python3
"""
Main script for building multilayer networks from single-cell data.

This script processes single-cell RNA-seq data and constructs multilayer networks
based on ligand-receptor interactions between specified cell types.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional, Set, List, Tuple

import anndata
import numpy as np
import torch
import pandas as pd

# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parents[1]))

from mlnetst.core.knowledge.networks import load_resource
from mlnetst.core.network.build_network import assemble_multilayer_network
from mlnetst.utils.computation_utils import compute_tensor_memory_usage
from mlnetst.utils.mlnet_logging import get_colored_logger

from mlnetst.utils.mlnet_utils import (
    build_supra_adjacency_matrix_from_tensor,
)

from mlnetst.utils.mlnet_metrics_utils import (
    compute_indegree,
    compute_instrength,
    compute_outdegree,
    compute_multi_indegree,
    compute_multi_outdegree,
    compute_total_degree,
    compute_clustering_coefficient
)

# Constants
RANDOM_STATE = 42
DEFAULT_DATA_PATH = Path(__file__).parents[1] / "data" / "processed" / "mouse1_slice153_x_hat_s.h5ad"


def setup_logging(verbose_level: int) -> Tuple[logging.Logger, str]:
    """
    Set up structured logging with appropriate levels.
    
    Args:
        verbose_level: Verbosity level (0=INFO, 1+=DEBUG)
        
    Returns:
        Tuple of (main_logger, log_mode)
    """
    if verbose_level >= 1:
        log_mode = "debug"
        main_level = logging.DEBUG
    else:
        log_mode = "info"
        main_level = logging.INFO
    
    # Create main logger
    main_logger = get_colored_logger("MAIN", level=main_level, mode=log_mode)
    
    return main_logger, log_mode

def str_or_none(value: str) -> Optional[str]:
    """
    Convert a string to None if it is 'None', otherwise return the string.
    
    Args:
        value: Input string
        
    Returns:
        str or None
    """
    return value if value.lower() != "none" else None

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build multilayer networks from single-cell data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
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
    
    parser.add_argument(
        "--mode", "-m", 
        type=str, 
        default="tensor", 
        choices=["tensor", "pymnet_multinetwork"],
        help="Mode to build the network"
    )
    
    parser.add_argument(
        "--resource", "-r", 
        type=str, 
        default="nichenet",
        choices=["nichenet", "mouseconsensus"],
        help="Resource to use for ligand-receptor interactions"
    )
    
    parser.add_argument(
        "--verbose", "-v", 
        type=int, 
        default=0, 
        help="Verbosity level: 0 for INFO, 1+ for DEBUG"
    )
    
    parser.add_argument(
        "--source_cell_type", 
        type=str_or_none,
        default=None,
        help="Source cell type for analysis (use 'None' for all cell types)"
    )

    parser.add_argument(
        "--target_cell_type", 
        type=str_or_none,
        default=None,
        help="Target cell type for analysis (use 'None' for all cell types)"
    )
    
    parser.add_argument(
        "--data_path", 
        type=Path, 
        default=DEFAULT_DATA_PATH,
        help="Path to the input AnnData file"
    )
    
    parser.add_argument(
        "--force", 
        action="store_true", 
        default=False,
        help="Force overwrite of existing multilayer network file"
    )
    
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads to use for parallel processing"
    )
    
    return parser.parse_args()


def load_and_filter_data(
    data_path: Path, 
    source_type: str, 
    target_type: str,
    logger: logging.Logger
) -> anndata.AnnData:
    """
    Load and filter single-cell data for specified cell types.
    
    Args:
        data_path: Path to the AnnData file
        source_type: Source cell type name
        target_type: Target cell type name
        logger: Logger instance
        
    Returns:
        Filtered AnnData object
    """
    logger.info(f"Loading data from: {data_path}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    x_hat_s = anndata.read_h5ad(data_path)
    logger.info(f"Loaded data with shape: {x_hat_s.shape}")
    
    # Filter for specified cell types
    if source_type is None or target_type is None:
        cell_types = x_hat_s.obs["subclass"].unique()
        logger.warning(
            "Source or target cell type not specified. Using all available cell types."
        )
        subdata = x_hat_s
    else:
        cell_types = [source_type, target_type]
        subdata = x_hat_s[x_hat_s.obs["subclass"].isin(cell_types), :]
    
    logger.info(f"Filtered data for cell types {cell_types}: {subdata.shape}")
    logger.debug(f"Cell type distribution:\n{subdata.obs['subclass'].value_counts()}")
    
    return subdata


def sample_cells(
    subdata: anndata.AnnData, 
    num_cells: int,
    logger: logging.Logger
) -> List[str]:
    """
    Sample cells from the dataset.
    
    Args:
        subdata: AnnData object to sample from
        num_cells: Number of cells to sample
        logger: Logger instance
        
    Returns:
        List of selected cell indices
    """
    total_cells = len(subdata.obs_names)
    
    if num_cells == total_cells:
        logger.info(f"Using all {total_cells} available cells")
        return subdata.obs_names.tolist()
    elif num_cells > total_cells:
        logger.warning(f"Requested {num_cells} cells but only {total_cells} available. Using all cells.")
        return subdata.obs_names.tolist()
    else:
        logger.info(f"Sampling {num_cells} cells from {total_cells} available")
        sampled_cells = subdata.obs.sample(num_cells, replace=False, random_state=RANDOM_STATE).index.tolist()
        return sampled_cells


def is_valid_complex(gene_string: str, valid_genes: Set[str]) -> bool:
    """
    Check if all components of a gene complex are present in valid genes.
    
    Args:
        gene_string: Gene or complex string (components separated by '_')
        valid_genes: Set of valid gene names (lowercase)
        
    Returns:
        True if all components are valid
    """
    components = gene_string.split("_")
    return all(comp.lower() in valid_genes for comp in components)


def filter_lr_interactions(
    subdata: anndata.AnnData, 
    num_layers: int,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Load and filter ligand-receptor interactions.
    
    Args:
        subdata: AnnData object containing gene information
        num_layers: Required number of interactions
        logger: Logger instance
        
    Returns:
        Filtered DataFrame of ligand-receptor interactions
    """
    logger.info("Loading ligand-receptor interaction database")
    lr_interactions_df = load_resource("mouseconsensus")
    logger.debug(f"Loaded {len(lr_interactions_df)} total interactions")
    
    # Create set of valid genes for efficient lookup
    valid_genes = set(g.lower() for g in subdata.var_names)
    logger.debug(f"Found {len(valid_genes)} valid genes in dataset")
    
    # Filter interactions where both source and target genes are valid
    logger.info("Filtering interactions for valid gene complexes")
    filtered_df = lr_interactions_df[
        lr_interactions_df["source"].apply(lambda x: is_valid_complex(x, valid_genes)) &
        lr_interactions_df["target"].apply(lambda x: is_valid_complex(x, valid_genes))
    ]
    
    logger.info(f"Found {len(filtered_df)} valid interactions after filtering")
    
    if len(filtered_df) < num_layers:
        logger.warning(f"Requested {num_layers} layers but only {len(filtered_df)} valid interactions available. Sampling from available interactions.")
        return filtered_df
    else:
        # Sample required number of interactions
        sample_lr = filtered_df.sample(n=num_layers, random_state=RANDOM_STATE).reset_index(drop=True)
    logger.info(f"Sampled {len(sample_lr)} interactions for network layers")
    return sample_lr


def build_multilayer_network(
    subdata: anndata.AnnData,
    cell_indexes: List[str],
    lr_interactions: pd.DataFrame,
    resource: str,
    mode: str,
    logger: logging.Logger,
    verbose: int = 1,
    num_threads: int = 1
) -> object:
    """
    Build the multilayer network.
    
    Args:
        subdata: AnnData object with expression data
        cell_indexes: List of cell indices to use
        lr_interactions: DataFrame of ligand-receptor interactions
        mode: Network building mode
        logger: Logger instance
        
    Returns:
        Assembled multilayer network object
    """
    # Create subset of data with selected cells
    network_data = subdata[cell_indexes, :]
    
    logger.info("Building multilayer network layers:")
    layer_descriptions = [
        f"{source} -> {target}" 
        for source, target in zip(lr_interactions["source"], lr_interactions["target"])
    ]
    for i, desc in enumerate(layer_descriptions, 1):
        logger.debug(f"  Layer {i}: {desc}")
    
    # Compute memory usage estimate
    num_cells = len(cell_indexes)
    num_layers = len(lr_interactions)
    compute_tensor_memory_usage(num_cells, num_layers)
    
    # Create network-building logger with DEBUG level
    if verbose >= 1:
        network_logger = get_colored_logger("NETWORK_BUILD", level=logging.DEBUG, mode="debug")
    else:
        network_logger = get_colored_logger("NETWORK_BUILD", level=logging.INFO, mode="info")
    
    logger.info(f"Assembling multilayer network with {num_cells} cells and potentially {num_layers} layers")
    
    mlnet, layer_mapping = assemble_multilayer_network(
        data=network_data,
        lr_db=lr_interactions,
        resource=resource,
        batch_size=None,
        toll_complexes=1e-6,
        toll_distance=1e-6,
        build_intra=True,
        build_inter=True,
        logger=network_logger,
        mode=mode,
        n_jobs=num_threads,
    )
    
    logger.info("Successfully built multilayer network")
    return mlnet


def main() -> None:
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Set up logging
    logger, log_mode = setup_logging(args.verbose)
    
    # Set random seeds for reproducibility
    logger.info(f"Setting random seed to {RANDOM_STATE}")
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    
    logger.info("Starting multilayer network construction")
    logger.info(f"Parameters: N={args.num_cells}, L={args.num_layers}, mode={args.mode}")
    
    # Load and filter data
    subdata = load_and_filter_data(
        args.data_path, 
        args.source_cell_type, 
        args.target_cell_type,
        logger
    )
    
    # Filter ligand-receptor interactions
    lr_interactions = filter_lr_interactions(subdata, args.num_layers, logger)
    
    subdata.X[subdata.X < 25] = 0
    # Let's do some quality control over subdata
    import scanpy as sc
    logger.info("Performing quality control on the subset of data")
    sc.pp.calculate_qc_metrics(subdata, inplace=True)
    logger.info(f"Subset info:\n{subdata.obs.info()}")
    sc.pl.violin(subdata, ["n_genes_by_counts", "log1p_n_genes_by_counts", "total_counts", "log1p_total_counts", "pct_counts_in_top_50_genes"],
                jitter=0.4, multi_panel=True, show=True) if args.verbose >= 1 else None
    
    sc.pp.filter_cells(subdata, min_genes=100)
    sc.pp.filter_genes(subdata, min_cells=3)
    
    logger.info("Performing quality control on the subset of data after filtering")
    sc.pp.calculate_qc_metrics(subdata, inplace=True)
    logger.info(f"Subset info:\n{subdata.obs.info()}")
    sc.pl.violin(subdata, ["n_genes_by_counts", "log1p_n_genes_by_counts", "total_counts", "log1p_total_counts", "pct_counts_in_top_50_genes"],
                jitter=0.4, multi_panel=True, show=True) if args.verbose >= 1 else None
    
    # Sample cells
    if args.num_cells >= len(subdata.obs_names):
        logger.warning(f"Requested {args.num_cells} cells but only {len(subdata.obs_names)} available. Using all cells.")
        cell_indexes = subdata.obs_names.tolist()
    else:
        cell_indexes = sample_cells(subdata, args.num_cells, logger)

    if (Path(__file__).parents[1] / "data" / "processed" / "experiment_mlnet.pth").exists() and args.force is False:
        logger.info("Multilayer network already exists. Use --force to overwrite.")
        mlnet = torch.load(Path(__file__).parents[1] / "data" / "processed" / "experiment_mlnet.pth")
    else:
        try:
            # Build multilayer network
            mlnet, layer_mapping = build_multilayer_network(
                subdata, 
                cell_indexes, 
                lr_interactions,
                args.resource,
                args.mode, 
                logger,
                args.verbose,
                num_threads=args.num_threads
            )
            
            logger.info("✅ Multilayer network construction completed successfully")
            
            # TODO: Add network analysis, saving, or visualization here
            
        except Exception as e:
            logger.error(f"❌ Error during execution: {str(e)}")
            logger.debug("Full traceback:", exc_info=True)
            sys.exit(1)
            
        torch.save(mlnet, Path(__file__).parents[1] / "data" / "processed" / "experiment_mlnet.pth")
        # save layer_mapping, which is a dictionary
        pd.DataFrame.from_dict(layer_mapping, orient="index").to_csv(
            Path(__file__).parents[1] / "data" / "processed" / "experiment_layer_mapping.csv"
        )
        logger.info("Multilayer network saved to disk")

    # Compute degree of the network
    supra_adjacency_matrix = build_supra_adjacency_matrix_from_tensor(mlnet)
    in_strength_distribution = compute_instrength(supra_adjacency_matrix, args.num_cells, args.num_layers)
    in_degree_distribution = compute_indegree(supra_adjacency_matrix, args.num_cells, args.num_layers)
    df_to_plot = pd.DataFrame({"value": in_strength_distribution.cpu().numpy(), "metric": "instrength"})
    df_to_plot = pd.concat([df_to_plot, pd.DataFrame({"value": in_degree_distribution.cpu().numpy(), "metric": "indegree"})], ignore_index=True)
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.catplot(df_to_plot, x="value", kind="violin", col="metric")
    plt.show()
    
    import networkx as nx
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    # Create graph
    # Your existing graph setup
    g = nx.Graph()
    nodes_dict = {
        i: {
            "x": subdata.obs.loc[i, "centroid_x"],
            "y": subdata.obs.loc[i, "centroid_y"],
            "instrength": float(in_strength_distribution[counter]),
            "indegree": float(in_degree_distribution[counter])
        } for counter, i in enumerate(cell_indexes)
    }
    g.add_nodes_from(nodes_dict.items())

    # Create figure
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    pos = {i: (nodes_dict[i]["x"], nodes_dict[i]["y"]) for i in nodes_dict}

    # Get color values
    instrength_values = [nodes_dict[i]["instrength"] for i in nodes_dict]
    indegree_values = [nodes_dict[i]["indegree"] for i in nodes_dict]

    # Print some statistics to check if values are actually different
    print("In-strength stats:")
    print(f"  Min: {min(instrength_values):.3f}, Max: {max(instrength_values):.3f}")
    print(f"  Mean: {np.mean(instrength_values):.3f}, Std: {np.std(instrength_values):.3f}")

    print("In-degree stats:")
    print(f"  Min: {min(indegree_values):.3f}, Max: {max(indegree_values):.3f}")
    print(f"  Mean: {np.mean(indegree_values):.3f}, Std: {np.std(indegree_values):.3f}")

    # Create normalizers - use same normalization range for better comparison
    global_min = min(min(instrength_values), min(indegree_values))
    global_max = max(max(instrength_values), max(indegree_values))
    global_norm = Normalize(vmin=global_min, vmax=global_max)

    # Individual normalizers (for separate color scales)
    instrength_norm = Normalize(vmin=min(instrength_values), vmax=max(instrength_values))
    indegree_norm = Normalize(vmin=min(indegree_values), vmax=max(indegree_values))

    # Plot 1: In-strength with individual normalization
    nx.draw(g, pos, node_size=5, with_labels=False,
            node_color=instrength_values,
            cmap=plt.cm.viridis,
            ax=axs[0])
    axs[0].set_title("In-strength (individual scale)")
    plt.colorbar(plt.cm.ScalarMappable(norm=instrength_norm, cmap=plt.cm.viridis),
                ax=axs[0], label='In-strength')

    # Plot 2: In-degree with individual normalization  
    nx.draw(g, pos, node_size=5, with_labels=False,
            node_color=indegree_values,
            cmap=plt.cm.viridis,
            ax=axs[1])
    axs[1].set_title("In-degree (individual scale)")
    plt.colorbar(plt.cm.ScalarMappable(norm=indegree_norm, cmap=plt.cm.viridis),
                ax=axs[1], label='In-degree')

    # Plot 3: Side-by-side with same color scale for direct comparison
    # Use global normalization so colors are directly comparable
    nx.draw(g, pos, node_size=50, with_labels=False,
            node_color=instrength_values,
            cmap=plt.cm.viridis,
            ax=axs[2])
    # Overlay with different marker for indegree (optional)
    scatter = axs[2].scatter([pos[i][0] for i in nodes_dict], 
                            [pos[i][1] for i in nodes_dict],
                            c=indegree_values, cmap=plt.cm.plasma, 
                            s=20, alpha=0.7, marker='s')
    axs[2].set_title("In-strength (viridis) vs In-degree (plasma)")
    plt.colorbar(plt.cm.ScalarMappable(norm=global_norm, cmap=plt.cm.viridis),
                ax=axs[2], label='Values (global scale)')

    plt.tight_layout()
    plt.show()

    # Calculate correlation to check if they're actually similar
    correlation = np.corrcoef(instrength_values, indegree_values)[0, 1]
    print(f"\nCorrelation between in-strength and in-degree: {correlation:.3f}")

    # Create a scatter plot to visualize the relationship
    plt.figure(figsize=(8, 6))
    plt.scatter(instrength_values, indegree_values, alpha=0.6)
    plt.xlabel('In-strength')
    plt.ylabel('In-degree')
    plt.title(f'In-strength vs In-degree (correlation: {correlation:.3f})')
    plt.show()

if __name__ == "__main__":
    main()
