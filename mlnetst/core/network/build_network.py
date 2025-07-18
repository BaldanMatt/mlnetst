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

def build_sparse_multilayer_network(N,L, cell_indexes, sample_lr, data,
                                    toll_dist=1e-10, toll_geom_mean=1e-10,
                                    compute_intralayer=True, compute_interlayer=True,
                                    n_jobs=1, sparsity_threshold=1e-10, enable_logging=False):
    """
    Memory-efficient sparse multilayer network construction.
    Args:
        N, L: Network dimensions
        cell_indexes, sample_lr, data: Data inputs
        toll_dist, toll_geom_mean: Numerical tolerances
        compute_intralayer: If True, compute same-layer interactions (default: True)
        compute_interlayer: If True, compute cross-layer interactions (default: True)
        n_jobs: Number of threads for parallel processing (default: 1)
                -1 uses all available cores, 1 disables parallelization
    Returns:
        torch.sparse.FloatTensor: Sparse tensor in COO format (N, L, N, L)
    """
    def log_info(message):
        """Simple logger function"""
        if enable_logging:
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            print(f"[{timestamp}] {message}")
    
    tic = time.time()

    # Determine number of threads for parallelization
    if n_jobs == -1:
        n_jobs = os.cpu_count()
    elif n_jobs <= 0:
        n_jobs = 1

    log_info(f"Starting sparse multilayer network construction...")
    log_info(f"Configuration: {n_jobs} thread(s), Intralayer: {'Yes' if compute_intralayer else 'No'}, Interlayer: {'Yes' if compute_interlayer else 'No'}")

    # Pre extract IDS (minimal memory: 2L strings)
    ligand_ids = [sample_lr["source"].iloc[l].lower() for l in range(L)]
    receptor_ids = [sample_lr["target"].iloc[l].lower() for l in range(L)]

    #TODO: Change this function to be batch on cell_ids. Do not want to call it N times.
    def get_expression_value(cell_id, gene_id):
        """Memory efficient expression extraction"""
        if len(gene_id.split("_")) > 1:
            gene_list = gene_id.split("_")
            expr = torch.tensor(data[cell_id,gene_list].X.astype(np.float32))
            result = torch.exp(torch.mean(torch.log(expr + toll_geom_mean), dim=1)).item()
            del expr
            return result
        else:
            return float(data[cell_id, gene_id].X.flatten()[0])

    def get_expression_value_batch(cell_indices, gene_id):
        if len(gene_id.split("_")) > 1:
            gene_list = gene_id.split("_")
            expr_matrix = data[cell_indices,:][:, gene_list].X.astype(np.float32)
            if hasattr(expr_matrix, 'toarray'):
                expr_matrix = expr_matrix.toarray()
            expr_tensor = torch.tensor(expr_matrix, dtype=torch.float32)
            result = torch.exp(torch.mean(torch.log(expr_tensor + toll_geom_mean), dim=1))
            return result.squeeze()
        else:
            expr_vector = data[cell_indices, gene_id].X
            if hasattr(expr_vector, 'toarray'):
                expr_vector = expr_vector.toarray()
            else:
                expr_vector = expr_vector.flatten()
            return torch.tensor(expr_vector, dtype=torch.float32).squeeze()

    # Lists to store sparse tensor components
    indices_list = [] # Will store [dim0, dim1, dim2, dim3] indices
    values_list = [] #`Will store corresponding values

    # Distance matrix computation (only if computing intralayer)
    dist_matrix = None
    if compute_intralayer:
        log_info("Computing distance matrix...")
        coords_x = torch.tensor([data.obs.loc[cell_id, "centroid_x"] for cell_id in cell_indexes],
                                dtype=torch.float32)
        coords_y = torch.tensor([data.obs.loc[cell_id, "centroid_y"] for cell_id in cell_indexes],
                                dtype=torch.float32)
        dist_matrix = torch.sqrt((coords_x.unsqueeze(1) - coords_x.unsqueeze(0))**2 +
                                 (coords_y.unsqueeze(1) - coords_y.unsqueeze(0))**2) + toll_dist
        del coords_x, coords_y

    # Intralayer interactions: same layer (α == β), different cells (i ≠ j)
    if compute_intralayer:
        log_info("Computing intralayer interactions...")
        intralayer_start = time.time()
        completed_layers = 0

        def compute_intralayer_sparse_for_layer(alpha):
            """Compute sparse intralayer interactions for a single layer:"""
            ligand_id = ligand_ids[alpha]
            receptor_id = receptor_ids[alpha]
            # Extract expression vectors for this layer
            all_cell_indices = cell_indexes
            #ligand_vals = torch.tensor([get_expression_value(cell_indexes[i], ligand_id)
            #                            for i in range(N)], dtype=torch.float32)
            #receptor_vals = torch.tensor([get_expression_value(cell_indexes[j], receptor_id)
            #                              for j in range(N)], dtype=torch.float32)
            ligand_vals = get_expression_value_batch(all_cell_indices, ligand_id)
            receptor_vals = get_expression_value_batch(all_cell_indices, receptor_id)
            # Create interaction matrix using broadcasting
            #print(f"[DEBUG] {ligand_vals.shape} and  {receptor_vals.shape}")
            interaction_matrix = torch.outer(ligand_vals, receptor_vals) / dist_matrix
            interaction_matrix.fill_diagonal_(0)  # Remove self-interactions
            # Get non-zero indices and values
            mask = interaction_matrix.abs() > sparsity_threshold
            nonzero_positions = torch.nonzero(mask, as_tuple=False)
            nonzero_values = interaction_matrix[mask]

            # Convert to 4D indices: [i, alpha, j, alpha]
            layer_indices = []
            layer_values = []
            if len(nonzero_positions)>0:
                # Create 4D indices for this layer
                i_indices = nonzero_positions[:, 0]
                j_indices = nonzero_positions[:, 1]
                alpha_indices = torch.full_like(i_indices, alpha)

                # Stack as [dim0, dim1, dim2, dim3] = [i, alpha, j, alpha]
                layer_indices = torch.stack([i_indices, alpha_indices, j_indices, alpha_indices])
                layer_values = nonzero_values
            del interaction_matrix, ligand_vals, receptor_vals
            return layer_indices, layer_values
        
        if n_jobs == 1:
            # Sequential processing
            for alpha in range(L):
                layer_indices, layer_values = compute_intralayer_sparse_for_layer(alpha)
                if len(layer_values)>0:
                    indices_list.append(layer_indices)
                    values_list.append(layer_values)
                completed_layers += 1
                
                if enable_logging and (completed_layers % max(1, L//10) == 0 or completed_layers == L):

                    elapsed = time.time() - intralayer_start
                    progress_pct = (completed_layers / L) * 100
                    eta = (elapsed / completed_layers) * (L - completed_layers) if completed_layers < L else 0
                    log_info(f"Intralayer progress {completed_layers}/{L} layers: {progress_pct:.1f}%, ETA: {eta:.2f} s")

        else:
        # parallel processing over layers
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all layer computations
                futures = {
                        executor.submit(compute_intralayer_sparse_for_layer, alpha): alpha
                        for alpha in range(L)
                        }
                # Collect results as they complete
                for future in as_completed(futures):
                    layer_indices, layer_values = future.result()
                    if len(layer_values)>0:
                        indices_list.append(layer_indices)
                        values_list.append(layer_values)
                    completed_layers += 1

                    if enable_logging and (completed_layers % max(1, L//10) == 0 or completed_layers == L):
                        elapsed = time.time() - intralayer_start
                        progress_pct = (completed_layers / L) * 100
                        eta = (elapsed / completed_layers) * (L - completed_layers) if completed_layers < L else 0
                        log_info(f"Intralayer progress: {completed_layers}/{L} layers ({progress_pct:.1f}%) - ETA: {eta:.1f}s")

        intralayer_time = time.time() - intralayer_start
        log_info(f"Intralayer interactions computed in {intralayer_time:.2f} s")

    # Interlayer interactions: different layers (α ≠ β), same cells (i == j)
    if compute_interlayer:
        # Generate all layer pairs (alpha != beta)
        layer_pairs = [(alpha, beta) for alpha in range(L) for beta in range(L) if alpha != beta]
        total_pairs = len(layer_pairs)
        log_info(f"Computing interlayer interactions for {total_pairs} layer pairs...")
        interlayer_start = time.time()
        completed_pairs = 0

        def compute_interlayer_sparse_for_pair(alpha_beta_pair):
            """ Compute sparse interlayer interactions for a layer pair."""
            alpha, beta = alpha_beta_pair
            receptor_alpha_id = receptor_ids[alpha]
            ligand_beta_id = ligand_ids[beta]

            # Extract expressions for all cells
            #TODO: Update once get_expression_value is batch compatible. 
            all_cell_indices = cell_indexes
            receptor_alpha_vals = get_expression_value_batch(all_cell_indices, receptor_alpha_id)
            ligand_beta_vals = get_expression_value_batch(all_cell_indices, ligand_beta_id)
            
            

            #receptor_alpha_vals = torch.tensor([get_expression_value(cell_indexes[i], receptor_alpha_id)
            #                                    for i in range(N)], dtype=torch.float32)
            #ligand_beta_vals = torch.tensor([get_expression_value(cell_indexes[i], ligand_beta_id)
            #                                for i in range(N)], dtype=torch.float32)

            # Element-wise multiplication for diagonal elements
            diagonal_values = receptor_alpha_vals * ligand_beta_vals
            # Get non-zero indices and values
            mask = diagonal_values.abs() > sparsity_threshold
            nonzero_positions = torch.nonzero(mask, as_tuple=False).squeeze()
            if nonzero_positions.numel() == 0:
                return torch.empty((4,0), dtype=torch.long), torch.empty(0, dtype=torch.float32)

            nonzero_values = diagonal_values[mask]

            # Convert to 4D indices: [i, alpha, i, beta]
            pair_indices = []
            pair_values = []
            if len(nonzero_values)>0:
                cell_indices = nonzero_positions
                alpha_indices = torch.full_like(cell_indices, alpha)
                beta_indices = torch.full_like(cell_indices, beta)
                # Stack as [dim0, dim1, dim2, dim3] = [i, alpha, i, beta]
                pair_indices = torch.stack([cell_indices, alpha_indices, cell_indices, beta_indices])
                pair_values = nonzero_values

            del diagonal_values, receptor_alpha_vals, ligand_beta_vals
            return pair_indices, pair_values

        if n_jobs == 1:
            # Sequential processing
            for alpha, beta in layer_pairs:
                pair_indices, pair_values = compute_interlayer_sparse_for_pair((alpha, beta))
                if len(pair_values)>0:
                    indices_list.append(pair_indices)
                    values_list.append(pair_values)

                completed_pairs += 1
                if enable_logging and (completed_pairs % max(1, total_pairs//10) == 0 or completed_pairs == total_pairs):
                    elapsed = time.time() - interlayer_start
                    progress_pct = (completed_pairs / total_pairs) * 100
                    eta = (elapsed / completed_pairs) * (total_pairs - completed_pairs) if completed_pairs < total_pairs else 0
                    log_info(f"Interlayer progress {completed_pairs}/{total_pairs} pairs: {progress_pct:.1f}%, ETA: {eta:.2f} s")
        else:
            # Parallel procesisng over layer pairs
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                # Submit all layer pair computations
                futures = {
                    executor.submit(compute_interlayer_sparse_for_pair, pair): pair
                    for pair in layer_pairs
                }
                # Collect results as they complete
                for future in as_completed(futures):
                    pair_indices, pair_values = future.result()
                    if len(pair_values)>0:
                        indices_list.append(pair_indices)
                        values_list.append(pair_values)
                    completed_pairs += 1
                    if enable_logging and (completed_pairs % max(1, total_pairs//10) == 0 or completed_pairs == total_pairs):
                        elapsed = time.time() - interlayer_start
                        progress_pct = (completed_pairs / total_pairs) * 100
                        eta = (elapsed / completed_pairs) * (total_pairs - completed_pairs) if completed_pairs < total_pairs else 0
                        log_info(f"Interlayer progress: {completed_pairs}/{total_pairs} pairs ({progress_pct:.1f}%) - ETA: {eta:.1f}s")
        interlayer_time = time.time() - interlayer_start
        log_info(f"Interlayer interactions computed in {interlayer_time:.2f} s")



    # Clean up distance matrix if it was created
    if dist_matrix is not None:
        del dist_matrix
    log_info("Assembling sparse tensor...")
    # Combine all indices and values
    if indices_list:
        # Concatenate all indices and values
        all_indices = torch.cat(indices_list, dim=1) # Shape: [4, total_nonzero_elements]
        all_values = torch.cat(values_list, dim=0) # Shape: [total_nonzero_elements]

        # Create sparse tensor in COO format
        mlnet_sparse = torch.sparse_coo_tensor(
            indices=all_indices,
            values=all_values,
            size=(N, L, N, L),
            dtype=torch.float32
        )
        # Coalesce to merge any duplicate indices - Good practice from pytorch documentation
        mlnet_sparse = mlnet_sparse.coalesce()
        log_info(f"Sparse tensor shape: {mlnet_sparse.shape}, non-zero elements: {mlnet_sparse._nnz()}")

    else:
        # All zeros - Create empty sparse tensor
        log_info("No non-zero elements found, returning empty sparse tensor.")
        mlnet_sparse = torch.sparse_coo_tensor(
                indices = torch.zeros((4,0), dtype=torch.long),
                values = torch.zeros(0, dtype=torch.float32),
                size=(N, L, N, L),
                dtype=torch.float32
        )
    toc = time.time()
    total_time = toc - tic
    log_info(f"Sparse network construction time: {total_time:.2f} s")
     # Memory usage comparison
    total_elements = N * L * N * L
    nonzero_elements = mlnet_sparse._nnz()
    sparsity = (1 - nonzero_elements / total_elements) * 100
    
    # Estimated memory usage
    dense_memory_gb = total_elements * 4 / (1024**3)  # 4 bytes per float32
    sparse_memory_gb = (nonzero_elements * 4 + nonzero_elements * 4 * 4) / (1024**3)  # values + indices
    log_info(f"Dense tensor would use: {dense_memory_gb:.2f} GB, sparse tensor uses: {sparse_memory_gb:.2f} GB, savings: {((dense_memory_gb - sparse_memory_gb) / dense_memory_gb * 100):.1f}%, sparsity: {sparsity:.2f}% ({nonzero_elements:,} / {total_elements:,} non-zero)")
    
    return mlnet_sparse


def build_multilayer_network(N, L, cell_indexes, sample_lr, data, 
                                            toll_dist=1e-10, toll_geom_mean=1e-10,
                                            compute_intralayer=True, compute_interlayer=True,
                                            n_jobs=1, enable_logging=False):
    """
    Memory-efficient fully vectorized multilayer network construction.
    
    Args:
        N, L: Network dimensions
        cell_indexes, sample_lr, data: Data inputs
        toll_dist, toll_geom_mean: Numerical tolerances
        compute_intralayer: If True, compute same-layer interactions (default: True)
        compute_interlayer: If True, compute cross-layer interactions (default: True)
        n_jobs: Number of threads for parallel processing (default: 1)
                -1 uses all available cores, 1 disables parallelization
    
    Memory Analysis:
    - Main tensor: N × L × N × L × 4 bytes ≈ 9.3 GB for N=1000, L=50
    - Can reduce memory by computing only intralayer OR interlayer interactions
    
    Parallelization Strategy:
    - Intralayer: Parallel over layers (L independent computations)
    - Interlayer: Parallel over layer pairs ((L choose 2) independent computations)
    - Expression extraction: Parallel over cells (N independent extractions)
    """

    def log_info(message):
        """Simple logger function"""
        if enable_logging:
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            print(f"[{timestamp}] {message}")
    
    tic = time.time()
    log_info(f"Starting multilayer network construction...")
    # Determine number of threads for parallelization
    if n_jobs == -1:
        import os
        n_jobs = os.cpu_count()
    elif n_jobs <= 0:
        n_jobs = 1
    
    log_info(f"Configuration: {n_jobs} thread(s), Intralayer: {'Yes' if compute_intralayer else 'No'}, Interlayer: {'Yes' if compute_interlayer else 'No'}")

    # Main tensor - this is our largest memory allocation
    log_info(f"Allocating main tensor ({N}×{L}×{N}×{L}) - estimated memory: {N*L*N*L*4/(1024**3):.2f} GB")

    # Main tensor - this is our largest memory allocation
    mlnet = torch.zeros(N, L, N, L, dtype=torch.float32)
    
    # Pre-extract IDs (minimal memory: 2L strings)
    log_info("Pre-extracting ligand and receptor IDs...")
    ligand_ids = [sample_lr["source"].iloc[l].lower() for l in range(L)]
    receptor_ids = [sample_lr["target"].iloc[l].lower() for l in range(L)]
    
    def get_expression_value(cell_id, gene_id):
        """Memory-efficient expression extraction."""
        if len(gene_id.split("_")) > 1:
            gene_list = gene_id.split("_")
            expr = torch.tensor(data[cell_id, gene_list].X.astype(np.float32))
            result = torch.exp(torch.mean(torch.log(expr + toll_geom_mean), dim=1)).item()
            del expr
            return result
        else:
            return float(data[cell_id, gene_id].X.flatten()[0])
    
    # Distance matrix computation (only if computing intralayer)
    dist_matrix = None
    if compute_intralayer:
        coords_x = torch.tensor([data.obs.loc[cell_id, "centroid_x"] for cell_id in cell_indexes], 
                               dtype=torch.float32)
        coords_y = torch.tensor([data.obs.loc[cell_id, "centroid_y"] for cell_id in cell_indexes], 
                               dtype=torch.float32)
        
        dist_matrix = torch.sqrt((coords_x.unsqueeze(1) - coords_x.unsqueeze(0))**2 + 
                               (coords_y.unsqueeze(1) - coords_y.unsqueeze(0))**2) + toll_dist
        del coords_x, coords_y
    
    # INTRALAYER INTERACTIONS: same layer (α == β), different cells (i ≠ j)
    if compute_intralayer:
        log_info("Computing intralayer interactions...")
        intralayer_start = time.time()
        completed_layers = 0
        def compute_intralayer_for_layer(alpha):
            """Compute intralayer interactions for a single layer."""
            ligand_id = ligand_ids[alpha]
            receptor_id = receptor_ids[alpha]
            
            # Extract expression vectors for this layer
            ligand_vals = torch.tensor([get_expression_value(cell_indexes[i], ligand_id) 
                                      for i in range(N)], dtype=torch.float32)
            receptor_vals = torch.tensor([get_expression_value(cell_indexes[j], receptor_id) 
                                        for j in range(N)], dtype=torch.float32)
            
            # Create interaction matrix using broadcasting
            interaction_matrix = torch.outer(ligand_vals, receptor_vals) / dist_matrix
            interaction_matrix.fill_diagonal_(0)  # Remove self-interactions
            
            return alpha, interaction_matrix
        
        if n_jobs == 1:
            # Sequential processing
            for alpha in range(L):
                _, interaction_matrix = compute_intralayer_for_layer(alpha)
                mlnet[:, alpha, :, alpha] = interaction_matrix
                del interaction_matrix
                completed_layers += 1

                if enable_logging and (completed_layers % max(1, L//10) == 0 or completed_layers == L):
                    elapsed = time.time() - intralayer_start
                    progress_pct = (completed_layers / L) * 100
                    eta = (elapsed / completed_layers) * (L - completed_layers) if completed_layers < L else 0
                    log_info(f"Intralayer progress {completed_layers}/{L} layers: {progress_pct:.1f}%, ETA: {eta:.2f} s")
        else:
            # Parallel processing over layers
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                # Submit all layer computations
                futures = {executor.submit(compute_intralayer_for_layer, alpha): alpha 
                          for alpha in range(L)}
                
                # Collect results as they complete
                for future in as_completed(futures):
                    alpha, interaction_matrix = future.result()
                    mlnet[:, alpha, :, alpha] = interaction_matrix
                    del interaction_matrix
                    completed_layers += 1
                    if enable_logging and (completed_layers % max(1, L//10) == 0 or completed_layers == L):
                        elapsed = time.time() - intralayer_start
                        progress_pct = (completed_layers / L) * 100
                        eta = (elapsed / completed_layers) * (L - completed_layers) if completed_layers < L else 0
                        log_info(f"Intralayer progress: {completed_layers}/{L} layers ({progress_pct:.1f}%) - ETA: {eta:.1f}s")
        intralayer_time = time.time() - intralayer_start
        log_info(f"Intralayer interactions computed in {intralayer_time:.2f} s")
    # INTERLAYER INTERACTIONS: different layers (α ≠ β), same cells (i == j)
    if compute_interlayer:
        # Generate all layer pairs (alpha != beta)
        layer_pairs = [(alpha, beta) for alpha in range(L) for beta in range(L) if alpha != beta]
        total_pairs = len(layer_pairs)
        log_info(f"Computing interlayer interactions for {total_pairs} layer pairs...")
        interlayer_start = time.time()
        completed_pairs = 0

        def compute_interlayer_for_pair(alpha_beta_pair):
            """Compute interlayer interactions for a layer pair."""
            alpha, beta = alpha_beta_pair
            receptor_alpha_id = receptor_ids[alpha]
            ligand_beta_id = ligand_ids[beta]
            
            # Extract expressions for all cells
            receptor_alpha_vals = torch.tensor([get_expression_value(cell_indexes[i], receptor_alpha_id) 
                                              for i in range(N)], dtype=torch.float32)
            ligand_beta_vals = torch.tensor([get_expression_value(cell_indexes[i], ligand_beta_id) 
                                           for i in range(N)], dtype=torch.float32)
            
            # Element-wise multiplication for diagonal elements
            diagonal_values = receptor_alpha_vals * ligand_beta_vals
            
            return (alpha, beta), diagonal_values
        
        
        if n_jobs == 1:
            # Sequential processing
            for alpha, beta in layer_pairs:
                _, diagonal_values = compute_interlayer_for_pair((alpha, beta))
                cell_indices = torch.arange(N)
                mlnet[cell_indices, alpha, cell_indices, beta] = diagonal_values
                del diagonal_values
                completed_pairs += 1

                if enable_logging and (completed_pairs % max(1, total_pairs//10) == 0 or completed_pairs == total_pairs):
                    elapsed = time.time() - interlayer_start
                    progress_pct = (completed_pairs / total_pairs) * 100
                    eta = (elapsed / completed_pairs) * (total_pairs - completed_pairs) if completed_pairs < total_pairs else 0
                    log_info(f"Interlayer progress {completed_pairs}/{total_pairs} pairs: {progress_pct:.1f}%, ETA: {eta:.2f} s")
        else:
            # Parallel processing over layer pairs
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                # Submit all layer pair computations
                futures = {executor.submit(compute_interlayer_for_pair, pair): pair 
                          for pair in layer_pairs}
                
                # Collect results as they complete
                cell_indices = torch.arange(N)  # Pre-compute indices
                for future in as_completed(futures):
                    (alpha, beta), diagonal_values = future.result()
                    mlnet[cell_indices, alpha, cell_indices, beta] = diagonal_values
                    del diagonal_values
                    completed_pairs += 1

                    if enable_logging and (completed_pairs % max(1, total_pairs//10) == 0 or completed_pairs == total_pairs):
                        elapsed = time.time() - interlayer_start
                        progress_pct = (completed_pairs / total_pairs) * 100
                        eta = (elapsed / completed_pairs) * (total_pairs - completed_pairs) if completed_pairs < total_pairs else 0
                        log_info(f"Interlayer progress: {completed_pairs}/{total_pairs} pairs ({progress_pct:.1f}%) - ETA: {eta:.1f}s")

        interlayer_time = time.time() - interlayer_start
        log_info(f"Interlayer interactions computed in {interlayer_time:.2f} s")
    
    # Clean up distance matrix if it was created
    if dist_matrix is not None:
        del dist_matrix

    toc = time.time()
    total_time = toc - tic
    log_info(f"Multilayer network construction completed in {total_time:.2f} s")
   
    if enable_logging:
        total_elements = N * L * N * L
        nonzero_elements = torch.count_nonzero(mlnet).item()
        sparsity = (1 - nonzero_elements / total_elements) * 100
        log_info(f"{nonzero_elements} non zero elements out of {total_elements:,} total elements ({sparsity:.2f}% sparsity)")
    return mlnet



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
    #print(f"Found {len(filtered_df)} valid interactions out of {len(lr_interactions_df)} total interactions.")
    # Check if we have enough interactions to sample
    #print(f"Unique sources: {filtered_df['source'].nunique()}, unique targets: {filtered_df['target'].nunique()}")
    # 4. Sample from filtered interactions
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


    old_mlnet = build_multilayer_network(N, L, cell_indexes, new_sample_lr, new_subdata,
                                         toll_dist=1e-6, toll_geom_mean=1e-6,
                                   compute_intralayer=True, compute_interlayer=True,
                                   n_jobs=1, enable_logging=True)
    # old_mlnet = build_sparse_multilayer_network(N, L, cell_indexes, new_sample_lr, new_subdata,
    #                                  compute_intralayer=True, compute_interlayer=True,
    #                                  toll_dist=1e-6, toll_geom_mean=1e-6,
    #                                  n_jobs=1, sparsity_threshold=0, enable_logging = True)

    # Check if the network are built coorectly
    result = torch.equal(new_mlnet.to_dense(), old_mlnet.to_dense())
    if not result:
       print("❌ Error: Dense and sparse networks are not equal!")
    else:
       print("✅ Dense and sparse networks are equal.")
    print("Multilayer network built successfully.")
    print(f"Network shape: {new_mlnet.shape}")
    
    


    #torch.save(mlnet, Path(__file__).parents[3] / "data" / "processed" / "mouse1_slice153_mlnet.pt")

