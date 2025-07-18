from pathlib import Path
import os
import time
import torch
import numpy as np
import pandas as pd
import anndata as ad
from mlnetst.core.knowledge.networks import load_resource
from mlnetst.utils.sparse_utils import compute_distance_matrix, compute_intralayer_interactions, compute_interlayer_interactions, create_layer_gene_mapping, select_intra_layers, create_layer_gene_mapping
from mlnetst.utils import compute_tensor_memory_usage
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any
import logging

COLORS_FOR_LOGGER = {
    "INFO": "\033[92m",  # Green
    "WARNING": "\033[93m",  # Yellow
    "ERROR": "\033[91m",  # Red
    "DEBUG": "\033[94m",  # Blue
    "RESET": "\033[0m"  # Reset to default color
}

def get_colored_logger(name: str, mode: str | None = None, log_file: str | None = None) -> logging.Logger:
    """
    Create a logger with specified name and optional file output.
    
    Args:
        name: Name of the logger
        mode: Logging level mode ('debug' or 'info')
        log_file: Optional file to log messages to
    
    Returns:
        Configured logger instance with no propagation to parent loggers
    """
    # Get or create logger
    logger = logging.getLogger(name)
    
    # Prevent propagation to parent loggers
    logger.propagate = False
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Set logging level
    if mode == "debug":
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        if mode is not None and mode != "info":
            print(f"Unknown logging mode: {mode}. Defaulting to INFO.")

    # Custom formatter for colored output
    class ColoredFormatter(logging.Formatter):
        def format(self, record):
            # Get color code for level, default to RESET if not found
            color = COLORS_FOR_LOGGER.get(record.levelname, COLORS_FOR_LOGGER["RESET"])
            # Apply color to the entire message
            log_fmt = (f"{color}%(asctime)s - %(name)s - %(levelname)s - "
                      f"%(message)s{COLORS_FOR_LOGGER['RESET']}")
            formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
            return formatter.format(record)

    # Console handler setup
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if mode == "debug" else logging.INFO)
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)
    
    # File handler setup if requested
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG if mode == "debug" else logging.INFO)
        # Use non-colored formatter for file output
        file_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        file_handler.setFormatter(logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(file_handler)
    
    return logger

def assemble_multilayer_network(data, lr_db,
                                batch_size: int | None = None,
                                toll_complexes: float = 1e-6, toll_distance: float = 1e-6,
                                build_intra: bool =True, build_inter: bool =True,
                                logger: Any | None = None, n_jobs:int = 1,
                                ) -> Any:

    """
    This function builds a sparse tensor of 4D that contains the interactions between
    nodes in multiple layers.
    Args:
        - data: anndata.AnnData containing the spatial transcriptomics count table
        - lr_db: pd.DataFrame database containing ligand-receptor pairs
        - toll_complexes: tolerance for the computation of the complex score
        - build_intra: flag to build intralayer scores
        - build_inter: flag to build interlayer scores
        - logger: Logger Object, if provided verbose is activated
        - n_jobs: number of workers/threads to parallelize computation
    Returns:
        - torch.sparse.FloatTensor: Sparse tensor in COO format (N, L, N, L)
    """
    start_building_time = time.time()
    # Determine number of threads for parallelization
    if n_jobs == -1:
        n_jobs = os.cpu_count()
    elif n_jobs <= 0:
        n_jobs = 1

    if n_jobs > 1:
        logger.warning(f"Parallel processing is not working, fallback to sequential") if logger else None
        n_jobs = 1
    # Determine logger and verbose mode
    if logger is not None and isinstance(logger, logging.Logger):
        ...
    else:
        print("Verbose mode deactivated")

    # Prepare runtime variables
    num_observations = data.shape[0]
    num_layers = lr_db.shape[0]
    ligand_ids = lr_db["source"].str.lower().tolist()
    receptor_ids = lr_db["target"].str.lower().tolist()
    layer_map = create_layer_gene_mapping(ligand_ids, receptor_ids, data.var_names)
    if logger:
        logger.debug(f"Layer map created with {len(layer_map)} entries.\n\t{layer_map}")
    # Init indices and values lists
    indices_list = []  # Will store [dim0, dim1, dim2, dim3] indices
    values_list = []  # Will store corresponding values
    
    # Extract cell indexes
    cell_indexes = data.obs_names
    
    ## Compute distance matrix
    if build_intra:
        coords_x = data.obs.loc[cell_indexes, "centroid_x"].astype(np.float32).tolist()
        coords_y = data.obs.loc[cell_indexes, "centroid_y"].astype(np.float32).tolist()
        dist_matrix = compute_distance_matrix(cell_indexes, coords_x, coords_y, toll_distance)
        if logger:
            logger.info("Distance matrix computed.")
        
        # Intralayer interactions: same layer (α == β), different cells (i ≠ j)
        if logger:
            logger.info("Computing intralayer interactions...")
        intralayer_start = time.time()
        completed_layers = 0
        
        #TODO: Possibly add module to select which layers to compute that outputs an iterable of indexes
        if n_jobs == 1:
            # Sequential processing
            for alpha, layer_info in layer_map.items():
                logger.debug(f"Computing intralayer interactions for layer {alpha} with l-r pair {layer_info['ligand']['gene_id']} -> {layer_info['receptor']['gene_id']}") if logger else None
                layer_indexes, layer_values = compute_intralayer_interactions(
                    data[cell_indexes, :], # data for selected cells
                    dist_matrix, # distance matrix for selected cells
                    alpha, # layer index
                    layer_info, # layer information
                    toll_complexes # tolerance for complex interactions
                )
                logger.debug(f"Layer {alpha} computed with {len(layer_values)} non-zero values.") if logger else None
                if len(layer_values)>0:
                    indices_list.append(layer_indexes)
                    values_list.append(layer_values)
                completed_layers += 1
                if logger and (completed_layers % max(1, num_layers // 10) == 0 or completed_layers == num_layers):
                    elapsed = time.time() - intralayer_start
                    progress_pct = (completed_layers / num_layers) * 100
                    eta = (elapsed / completed_layers) * (num_layers - completed_layers) if completed_layers < num_layers else 0
                    logger.info(f"Intralayer progress {completed_layers}/{num_layers} layers: {progress_pct:.1f}%, ETA: {eta:.2f} s")
        else:
            with ProcessPoolExecutor(max_workers = n_jobs) as executor:
            # Submit all layer computations
                futures = {
                        executor.submit(compute_intralayer_interactions, data[cell_indexes, :], dist_matrix, alpha, layer_info, toll_complexes): alpha
                        for alpha, layer_info in layer_map.items()
                        }
                # Collect results as they complete
                for future in as_completed(futures):
                    layer_indices, layer_values = future.result()
                    if len(layer_values)>0:
                        indices_list.append(layer_indices)
                        values_list.append(layer_values)
                    completed_layers += 1
                    if logger and (completed_layers % max(1, num_layers // 10) == 0 or completed_layers == num_layers):
                        elapsed = time.time() - intralayer_start
                        progress_pct = (completed_layers / num_layers) * 100
                        eta = (elapsed / completed_layers) * (num_layers - completed_layers) if completed_layers < num_layers else 0
                        logger.info(f"Intralayer progress: {completed_layers}/{num_layers} layers ({progress_pct:.1f}%) - ETA: {eta:.1f}s")
        intralayer_time = time.time() - intralayer_start
        logger.info(f"All intralayer interactions computed in {intralayer_time:.2f} s")
        
    if build_inter:
        # Interlayer interactions: different layers (α ≠ β), same cells (i == j)
        if logger:
            logger.info("Computing interlayer interactions...")
        interlayer_start = time.time()
        completed_pairs = 0

        layer_pairs = [(alpha, beta) for alpha in layer_map.keys() for beta in layer_map.keys() if alpha != beta]

        if n_jobs == 1:
            # Sequential processing
            for alpha, beta in layer_pairs:
                logger.debug(f"Computing interlayer interactions for layer pair ({alpha}, {beta}) with r_alpha-l_beta {layer_map[alpha]['receptor']['gene_id']} -> {layer_map[beta]['ligand']['gene_id']}") if logger else None
                layer_indexes, layer_values = compute_interlayer_interactions(
                    data[cell_indexes, :], # data for selected cells
                    dist_matrix, # distance matrix for selected cells
                    alpha, # source layer index
                    beta, # destination layer index
                    layer_map[alpha], # source layer information
                    layer_map[beta], # destination layer information
                    toll_complexes # tolerance for complex interactions
                )
                logger.debug(f"Layer pair ({alpha}, {beta}) computed with {len(layer_values)} non-zero values.") if logger else None
                if len(layer_values)>0:
                    indices_list.append(layer_indexes)
                    values_list.append(layer_values)
                completed_pairs += 1
                if logger and (completed_pairs % max(1, len(layer_pairs) // 10) == 0 or completed_pairs == len(layer_pairs)):
                    elapsed = time.time() - interlayer_start
                    progress_pct = (completed_pairs / len(layer_pairs)) * 100
                    eta = (elapsed / completed_pairs) * (len(layer_pairs) - completed_pairs) if completed_pairs < len(layer_pairs) else 0
                    logger.info(f"Interlayer progress {completed_pairs}/{len(layer_pairs)} pairs: {progress_pct:.1f}%, ETA: {eta:.2f} s")
        else:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Submit all layer pair computations
                futures = {
                        executor.submit(compute_interlayer_interactions, data[cell_indexes, :], dist_matrix, alpha, beta, toll_complexes): (alpha, beta)
                        for alpha, beta in layer_pairs
                        }
                # Collect results as they complete
                for future in as_completed(futures):
                    layer_indices, layer_values = future.result()
                    if len(layer_values)>0:
                        indices_list.append(layer_indices)
                        values_list.append(layer_values)
                    completed_pairs += 1
                    if logger and (completed_pairs % max(1, len(layer_pairs) // 10) == 0 or completed_pairs == len(layer_pairs)):
                        elapsed = time.time() - interlayer_start
                        progress_pct = (completed_pairs / len(layer_pairs)) * 100
                        logger.info(f"Interlayer progress: {completed_pairs}/{len(layer_pairs)} pairs ({progress_pct:.1f}%) - ETA: {eta:.1f}s")
                    if completed_pairs == len(layer_pairs):
                        logger.info(f"All interlayer interactions computed in {elapsed:.2f} s")
        interlayer_time = time.time() - interlayer_start
        logger.info(f"All interlayer interactions computed in {interlayer_time:.2f} s")
        
    # Clean up distance matrix if it was created
    if 'dist_matrix' in locals():
        del dist_matrix
    logger.info("Assembling sparse tensor...")
    # Combine all indices and values
    if indices_list:
        # Concatenate all indices and values
        all_indices = torch.cat(indices_list, dim=1)  # Shape: [4, total_nonzero_elements]
        all_values = torch.cat(values_list, dim=0)  # Shape: [total_nonzero_elements]

        # Create sparse tensor in COO format
        mlnet_sparse = torch.sparse_coo_tensor(
            indices=all_indices,
            values=all_values,
            size=(num_observations, num_layers, num_observations, num_layers),
            dtype=torch.float32
        )
        # Coalesce to merge any duplicate indices - Good practice from pytorch documentation
        mlnet_sparse = mlnet_sparse.coalesce()
        logger.info(f"Sparse tensor shape: {mlnet_sparse.shape}, non-zero elements: {mlnet_sparse._nnz()}")
    else:
        # All zeros - Create empty sparse tensor
        logger.info("No non-zero elements found, returning empty sparse tensor.")
        mlnet_sparse = torch.sparse_coo_tensor(
                indices=torch.zeros((4, 0), dtype=torch.long),
                values=torch.zeros(0, dtype=torch.float32),
                size=(num_observations, num_layers, num_observations, num_layers),
                dtype=torch.float32
        )
    end_building_time = time.time()
    total_time = end_building_time - start_building_time
    logger.info(f"Sparse network construction time: {total_time:.2f} s")
    # Memory usage comparison
    total_elements = num_observations * num_layers * num_observations * num_layers
    nonzero_elements = mlnet_sparse._nnz()
    sparsity = (1 - nonzero_elements / total_elements) * 100
    # Estimated memory usage
    dense_memory_gb = total_elements * 4 / (1024**3)  # 4 bytes per float32
    sparse_memory_gb = (nonzero_elements * 4 + nonzero_elements * 4 * 4) / (1024**3)  # values + indices
    logger.info(f"Dense tensor would use: {dense_memory_gb:.2f} GB, sparse tensor uses: {sparse_memory_gb:.2f} GB, savings: {((dense_memory_gb - sparse_memory_gb) / dense_memory_gb * 100):.1f}%, sparsity: {sparsity:.2f}% ({nonzero_elements:,} / {total_elements:,} non-zero)")
    # Return the sparse tensor
    return mlnet_sparse

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_layers", "-L", type=int)
    parser.add_argument("--num_cells", "--N", type=int)
    args = parser.parse_args()
    print("DEBUGGING build_network.py")
    x_hat_s = ad.read_h5ad(Path(__file__).parents[3] / "data" / "processed" / "mouse1_slice153_x_hat_s.h5ad")

    print(x_hat_s)
    source, target = "Astro", "L2/3 IT"
    subdata = x_hat_s[x_hat_s.obs["subclass"].isin([source,target]), :]
    print(subdata)

    N = args.num_cells
    L = args.num_layers

    print(f"[WARN] Number of cells: {N}, number of layers: {L}")
    RANDOM_STATE = 42
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    if N == len(subdata.obs_names):
        cell_indexes = subdata.obs_names
    else:
        cell_indexes = subdata.obs.sample(N, replace = False).index
    notFound = True
    lr_interactions_df = load_resource("mouseconsensus")

    # Convert subdata.var_names to lowercase set for faster lookup
    valid_genes = set(g.lower() for g in subdata.var_names)

    # 2. Define a helper to check if all components of a complex are in valid_genes
    def is_valid_complex(gene_string):
        components = gene_string.split("_")
        return all(comp.lower() in valid_genes for comp in components)

    # 3. Apply filtering to both source and target
    filtered_df = lr_interactions_df[
    lr_interactions_df["source"].apply(is_valid_complex) &
        lr_interactions_df["target"].apply(is_valid_complex)
    ]

    if len(filtered_df) >= L:
        sample_lr = filtered_df.sample(n=L)
        notFound = False
    else:
        print(f"❌ Not enough valid interactions (found {len(filtered_df)}, needed {L})")
        sample_lr = None  # Or handle appropriately

    new_subdata = subdata[cell_indexes, :]
    new_sample_lr = sample_lr.copy()
    new_sample_lr = new_sample_lr.sample(L, replace=False).reset_index(drop=True)
    print(f"Building the following layers: ")
    print([x+"->"+y
        for x,y in zip(new_sample_lr["source"].tolist(), new_sample_lr["target"].tolist())])
    compute_tensor_memory_usage(N,L) 
    new_mlnet = assemble_multilayer_network(
        data=new_subdata,
        lr_db=new_sample_lr,
        batch_size=None,
        toll_complexes=1e-6,
        toll_distance=1e-6,
        build_intra=True,
        build_inter=True,
        logger=get_colored_logger("Build network", mode="debug"),
        n_jobs=1,
    )
