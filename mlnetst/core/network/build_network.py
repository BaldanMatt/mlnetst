from pathlib import Path
import os
import time
import torch
import numpy as np
import pandas as pd
import anndata as ad
from mlnetst.core.knowledge.networks import load_resource
from mlnetst.utils import compute_tensor_memory_usage
from concurrent.futures import ThreadPoolExecutor, as_completed


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
            ligand_vals = torch.tensor([get_expression_value(cell_indexes[i], ligand_id)
                                        for i in range(N)], dtype=torch.float32)
            receptor_vals = torch.tensor([get_expression_value(cell_indexes[j], receptor_id)
                                          for j in range(N)], dtype=torch.float32)
            # Create interaction matrix using broadcasting
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
            receptor_alpha_vals = torch.tensor([get_expression_value(cell_indexes[i], receptor_alpha_id)
                                                for i in range(N)], dtype=torch.float32)
            ligand_beta_vals = torch.tensor([get_expression_value(cell_indexes[i], ligand_beta_id)
                                            for i in range(N)], dtype=torch.float32)
            # Element-wise multiplication for diagonal elements
            diagonal_values = receptor_alpha_vals * ligand_beta_vals
            # Get non-zero indices and values
            mask = diagonal_values.abs() > sparsity_threshold
            nonzero_positions = torch.nonzero(mask, as_tuple=False).squeeze()
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
    print("DEBUGGING build_network.py")
    x_hat_s = ad.read_h5ad(Path(__file__).parents[3] / "data" / "processed" / "mouse1_slice153_x_hat_s.h5ad")

    print(x_hat_s)
    source, target = "Astro", "L2/3 IT"
    subdata = x_hat_s[x_hat_s.obs["subclass"].isin([source,target]), :]
    print(subdata)

    N = 100
    L = 100

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

    # 4. Sample from filtered interactions
    if len(filtered_df) >= L:
        sample_lr = filtered_df.sample(n=L)
        notFound = False
    else:
        print(f"❌ Not enough valid interactions (found {len(filtered_df)}, needed {L})")
        sample_lr = None  # Or handle appropriately
    
    print(f"Building the following layers: {sample_lr['source'].tolist()} -> {sample_lr['target'].tolist()}")
    compute_tensor_memory_usage(N,L) 
    mlnet = build_multilayer_network(N, L, cell_indexes, sample_lr, subdata,
                                     compute_intralayer=True, compute_interlayer=True,
                                     n_jobs=-1, enable_logging=True)
    mlnet_sparse = build_sparse_multilayer_network(N, L, cell_indexes, sample_lr, subdata,
                                     compute_intralayer=True, compute_interlayer=True,
                                     n_jobs=-1, sparsity_threshold=0, enable_logging = True)

    # Check if the network are built coorectly
    result = torch.equal(mlnet, mlnet_sparse.to_dense())
    if not result:
        print("❌ Error: Dense and sparse networks are not equal!")
    else:
        print("✅ Dense and sparse networks are equal.")
    print("Multilayer network built successfully.")
    print(f"Network shape: {mlnet.shape}")
