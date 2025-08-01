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
        raise ValueError(
            f"Not enough valid interactions found. "
            f"Required: {num_layers}, Available: {len(filtered_df)}"
        )
    
    # Sample required number of interactions
    sample_lr = filtered_df.sample(n=num_layers, random_state=RANDOM_STATE).reset_index(drop=True)
    logger.info(f"Sampled {len(sample_lr)} interactions for network layers")
    
    return sample_lr


def build_multilayer_network(
    subdata: anndata.AnnData,
    cell_indexes: List[str],
    lr_interactions: pd.DataFrame,
    mode: str,
    logger: logging.Logger
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
        logger.info(f"  Layer {i}: {desc}")
    
    # Compute memory usage estimate
    num_cells = len(cell_indexes)
    num_layers = len(lr_interactions)
    compute_tensor_memory_usage(num_cells, num_layers)
    
    # Create network-building logger with DEBUG level
    network_logger = get_colored_logger("NETWORK_BUILD", level=logging.DEBUG, mode="debug")
    
    logger.info(f"Assembling multilayer network with {num_cells} cells and {num_layers} layers")
    
    mlnet = assemble_multilayer_network(
        data=network_data,
        lr_db=lr_interactions,
        batch_size=None,
        toll_complexes=1e-6,
        toll_distance=1e-6,
        build_intra=True,
        build_inter=True,
        logger=network_logger,
        mode=mode,
        n_jobs=1,
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
    
    try:
        # Load and filter data
        subdata = load_and_filter_data(
            args.data_path, 
            args.source_cell_type, 
            args.target_cell_type,
            logger
        )
        
        # Sample cells
        cell_indexes = sample_cells(subdata, args.num_cells, logger)
        
        # Filter ligand-receptor interactions
        lr_interactions = filter_lr_interactions(subdata, args.num_layers, logger)
        
        # Build multilayer network
        mlnet = build_multilayer_network(
            subdata, 
            cell_indexes, 
            lr_interactions, 
            args.mode, 
            logger
        )
        
        logger.info("✅ Multilayer network construction completed successfully")
        
        # TODO: Add network analysis, saving, or visualization here
        
    except Exception as e:
        logger.error(f"❌ Error during execution: {str(e)}")
        logger.debug("Full traceback:", exc_info=True)
        sys.exit(1)

    # Compute degree of the network
    supra_adjacency_matrix = build_supra_adjacency_matrix_from_tensor(mlnet)
    in_strength_distribution = compute_instrength(supra_adjacency_matrix, args.num_cells, args.num_layers)

    import seaborn as sns
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 6))
    sns.histplot(in_strength_distribution.cpu().numpy(), bins=30, kde=True)
    plt.title("Instrength Distribution")
    plt.xlabel("Instrength")
    plt.ylabel("Frequency")
    plt.show()
    
    print(subdata.obs_names)
    import networkx as nx
    g = nx.Graph()
    nodes_dict = {i: {"x": subdata.obs.loc[i,"centroid_x"], "y": subdata.obs.loc[i,"centroid_y"], "instrength": in_strength_distribution[counter]} for counter, i in enumerate(cell_indexes)}
    g.add_nodes_from(nodes_dict.items())
    pos = {i: (nodes_dict[i]["x"], nodes_dict[i]["y"]) for i in nodes_dict}
    nx.draw(g, pos, node_size=10, with_labels=False, 
            node_color=[nodes_dict[i]["instrength"] for i in nodes_dict], cmap=plt.cm.viridis)
    plt.show()

if __name__ == "__main__":
    main()
