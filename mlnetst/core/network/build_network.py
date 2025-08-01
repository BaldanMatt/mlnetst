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

def assemble_multilayer_network(data, lr_db,
                                batch_size: int | None = None,
                                toll_complexes: float = 1e-6, toll_distance: float = 1e-6,
                                build_intra: bool =True, build_inter: bool =True,
                                mode: str = "tensor",
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
    if mode == "tensor":
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
            # Memory usage comparison
            total_elements = num_observations * num_layers * num_observations * num_layers
            nonzero_elements = mlnet_sparse._nnz()
            sparsity = (1 - nonzero_elements / total_elements) * 100
            # Estimated memory usage
            dense_memory_gb = total_elements * 4 / (1024**3)  # 4 bytes per float32
            sparse_memory_gb = (nonzero_elements * 4 + nonzero_elements * 4 * 4) / (1024**3)  # values + indices
            logger.info(f"Dense tensor would use: {dense_memory_gb:.2f} GB, sparse tensor uses: {sparse_memory_gb:.2f} GB, savings: {((dense_memory_gb - sparse_memory_gb) / dense_memory_gb * 100):.1f}%, sparsity: {sparsity:.2f}% ({nonzero_elements:,} / {total_elements:,} non-zero)")
            
    elif mode == "pymnet_multinetwork":
        import pymnet
        # Create a pymnet multiplex network
        mlnet_sparse = pymnet.MultilayerNetwork(aspects=1, directed=True)
        # Add nodes
        for i in range(num_observations):
            mlnet_sparse.add_node(i)
        # Add layers
        for alpha, layer_info in layer_map.items():
            mlnet_sparse.add_layer(layer_info["layer_name"])
            
        logger.debug(mlnet_sparse.get_layers())
        # Add edges
        for counter_set, set_of_edges in enumerate(indices_list):
            logger.debug("Adding edges for set %d", counter_set)
            sender_indexes, layer_sender_indices, receivers_indexes, receivers_layer_indexes = set_of_edges
            for counter_value, (i, alpha, j, beta) in enumerate(zip(sender_indexes, layer_sender_indices, receivers_indexes, receivers_layer_indexes)):
                alpha_name = layer_map[alpha.item()]["layer_name"]
                beta_name = layer_map[beta.item()]["layer_name"]
                logger.debug("Adding edge from %d %s to %d %s with value %f", i, alpha_name, j, beta_name, values_list[counter_set][counter_value])
                mlnet_sparse[i.item(), j.item(), alpha_name, beta_name] = values_list[counter_set][counter_value].item()
    
    elif mode == "pymnet_multiplex":
        import pymnet
        # Create a pymnet multiplex network
        raise NotImplementedError("pymnet_multiplex mode is not implemented yet. This mode is constraining \
                                  interlayer interactions to be automatically filled by the pymnet library.")
        
    else:
        raise ValueError(f"Unknown mode: {mode}. Supported modes are 'tensor', 'pymnet_multinetwork', and 'pymnet_multiplex'.")
            
    end_building_time = time.time()
    total_time = end_building_time - start_building_time
    logger.info(f"Sparse network construction time: {total_time:.2f} s")
    
    # Return the sparse tensor
    return mlnet_sparse