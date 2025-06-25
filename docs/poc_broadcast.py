import sys
from pathlib import Path
import os
sys.path.append(str(Path(__file__).parents[1]))
import pandas as pd
import anndata as ad
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import random
from mlnetst.core.knowledge.networks import load_resource
import time
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
pd.core.common.random_state(None)
random.seed(RANDOM_SEED)

def build_multilayer_network_vectorized(N, L, cell_indexes, sample_lr, subdata, 
                                       toll_dist=1e-10, toll_geom_mean=1e-10):
    """
    Vectorized construction of multilayer network tensor.
    
    Args:
        N: Number of nodes
        L: Number of layers
        cell_indexes: List of cell identifiers
        sample_lr: DataFrame with 'source' and 'target' columns for ligand-receptor pairs
        subdata: Data object containing expression and spatial information
        toll_dist: Distance tolerance
        toll_geom_mean: Geometric mean tolerance
    
    Returns:
        mlnet: 4D tensor of shape (N, L, N, L)
    """
    
    tic = time.time()
    
    # Initialize the tensor
    mlnet = torch.zeros(N, L, N, L, dtype=torch.float32)
    
    # Pre-extract all ligand and receptor IDs
    ligand_ids = [sample_lr["source"].iloc[l].lower() for l in range(L)]
    receptor_ids = [sample_lr["target"].iloc[l].lower() for l in range(L)]
    
    # Pre-compute distance matrix for all cell pairs
    coords_x = torch.tensor([subdata.obs.loc[cell_id, "centroid_x"] for cell_id in cell_indexes], dtype=torch.float32)
    coords_y = torch.tensor([subdata.obs.loc[cell_id, "centroid_y"] for cell_id in cell_indexes], dtype=torch.float32)
    
    # Broadcasting to get all pairwise distances: (N, N)
    dist_matrix = torch.sqrt((coords_x.unsqueeze(1) - coords_x.unsqueeze(0))**2 + 
                           (coords_y.unsqueeze(1) - coords_y.unsqueeze(0))**2) + toll_dist
    
    # Pre-extract all expression values for efficiency
    def get_expression_vector(cell_id, gene_id):
        """Extract expression vector for a gene, handling multi-gene cases."""
        if len(gene_id.split("_")) > 1:
            gene_list = gene_id.split("_")
            expr = torch.tensor(subdata[cell_id, gene_list].X.astype(np.float32))
            return torch.exp(torch.mean(torch.log(expr + toll_geom_mean), dim=1)).flatten()
        else:
            return torch.tensor(subdata[cell_id, gene_id].X.astype(np.float32).flatten())
    
    # Pre-compute all expression values: dict[cell_id][gene_id] = expression_tensor
    expr_cache = {}
    for i, cell_id in enumerate(cell_indexes):
        expr_cache[cell_id] = {}
        for gene_id in set(ligand_ids + receptor_ids):
            expr_cache[cell_id][gene_id] = get_expression_vector(cell_id, gene_id)
    
    # === INTRALAYER INTERACTIONS (alpha == beta, i != j) ===
    # Create masks for valid intralayer pairs
    alpha_indices = torch.arange(L)
    i_indices, j_indices = torch.meshgrid(torch.arange(N), torch.arange(N), indexing='ij')
    
    # Mask for intralayer: same layer, different cells
    intralayer_mask = i_indices != j_indices  # (N, N) boolean mask
    
    for alpha in range(L):
        # Get all valid (i,j) pairs for this layer
        valid_i, valid_j = torch.where(intralayer_mask)
        
        if len(valid_i) == 0:
            continue
            
        # Vectorized computation for this layer
        ligand_id = ligand_ids[alpha]
        receptor_id = receptor_ids[alpha]
        
        # Extract expression values for all valid pairs
        a_values = torch.stack([expr_cache[cell_indexes[i.item()]][ligand_id] for i in valid_i])
        b_values = torch.stack([expr_cache[cell_indexes[j.item()]][receptor_id] for j in valid_j])
        
        # Get distances for valid pairs
        d_values = dist_matrix[valid_i, valid_j]
        
        # Compute interactions
        interactions = (a_values * b_values) / d_values.unsqueeze(-1)
        
        # Assign to tensor
        mlnet[valid_i, alpha, valid_j, alpha] = interactions.squeeze()
    
    # === INTERLAYER INTERACTIONS (alpha != beta, i == j) ===
    # Create masks for valid interlayer pairs
    layer_pairs = [(alpha, beta) for alpha in range(L) for beta in range(L) if alpha != beta]
    
    for alpha, beta in layer_pairs:
        # For interlayer: same cell, different layers
        receptor_alpha_id = receptor_ids[alpha]
        ligand_beta_id = ligand_ids[beta]
        
        # Extract expression values for all cells
        a_values = torch.stack([expr_cache[cell_id][receptor_alpha_id] for cell_id in cell_indexes])
        b_values = torch.stack([expr_cache[cell_id][ligand_beta_id] for cell_id in cell_indexes])
        
        # Compute interactions (diagonal elements only)
        interactions = a_values * b_values
        
        # Assign to tensor diagonal
        cell_indices = torch.arange(N)
        mlnet[cell_indices, alpha, cell_indices, beta] = interactions.squeeze()
    
    toc = time.time()
    print(f"Vectorized time elapsed: {toc-tic:.2f} s")
    
    return mlnet


# Alternative even more vectorized approach (if memory allows)
def build_multilayer_network_fully_vectorized(N, L, cell_indexes, sample_lr, subdata, 
                                            toll_dist=1e-10, toll_geom_mean=1e-10):
    """
    Fully vectorized approach - faster but more memory intensive.
    """
    tic = time.time()
    
    # Initialize tensor
    mlnet = torch.zeros(N, L, N, L, dtype=torch.float32)
    
    # Pre-extract expression data into tensors
    ligand_ids = [sample_lr["source"].iloc[l].lower() for l in range(L)]
    receptor_ids = [sample_lr["target"].iloc[l].lower() for l in range(L)]
    
    def get_expression_matrix(gene_ids):
        """Get expression matrix for all cells and given genes."""
        expr_matrix = torch.zeros(N, len(gene_ids), dtype=torch.float32)
        for i, cell_id in enumerate(cell_indexes):
            for j, gene_id in enumerate(gene_ids):
                if len(gene_id.split("_")) > 1:
                    gene_list = gene_id.split("_")
                    expr = torch.tensor(subdata[cell_id, gene_list].X.astype(np.float32))
                    expr_matrix[i, j] = torch.exp(torch.mean(torch.log(expr + toll_geom_mean), dim=1)).flatten()
                else:
                    expr_matrix[i, j] = torch.tensor(subdata[cell_id, gene_id].X.astype(np.float32).flatten())
        return expr_matrix
    
    # Get expression matrices: (N, L)
    ligand_expr = get_expression_matrix(ligand_ids)  # (N, L)
    receptor_expr = get_expression_matrix(receptor_ids)  # (N, L)
    
    # Distance matrix
    coords_x = torch.tensor([subdata.obs.loc[cell_id, "centroid_x"] for cell_id in cell_indexes])
    coords_y = torch.tensor([subdata.obs.loc[cell_id, "centroid_y"] for cell_id in cell_indexes])
    dist_matrix = torch.sqrt((coords_x.unsqueeze(1) - coords_x.unsqueeze(0))**2 + 
                           (coords_y.unsqueeze(1) - coords_y.unsqueeze(0))**2) + toll_dist
    
    # Intralayer interactions: same layer (alpha == beta), different cells (i != j)
    for alpha in range(L):
        # Create mask: different cells only
        mask = ~torch.eye(N, dtype=torch.bool)
        
        # Compute interactions for all valid pairs
        ligand_vals = ligand_expr[:, alpha].unsqueeze(1)  # (N, 1)
        receptor_vals = receptor_expr[:, alpha].unsqueeze(0)  # (1, N)
        
        # Broadcasting: (N, 1) * (1, N) / (N, N) -> (N, N)
        interactions = (ligand_vals * receptor_vals) / dist_matrix
        
        # Apply mask and assign
        mlnet[:, alpha, :, alpha] = interactions * mask
    
    # Interlayer interactions: different layers (alpha != beta), same cells (i == j)
    for alpha in range(L):
        for beta in range(L):
            if alpha != beta:
                # Only diagonal elements (same cells)
                interactions = receptor_expr[:, alpha] * ligand_expr[:, beta]
                mlnet[torch.arange(N), alpha, torch.arange(N), beta] = interactions
    
    toc = time.time()
    print(f"Fully vectorized time elapsed: {toc-tic:.2f} s")
    
    return mlnet

def build_multilayer_network_fully_vectorized_v2(N, L, cell_indexes, sample_lr, subdata, 
                                            toll_dist=1e-10, toll_geom_mean=1e-10):
    """
    Memory-efficient fully vectorized multilayer network construction.
    
    """
    
    tic = time.time()
    
    # Main tensor - this is our largest memory allocation
    # Memory: N × L × N × L × 4 bytes ≈ 9.3 GB for N=1000, L=50
    mlnet = torch.zeros(N, L, N, L, dtype=torch.float32)
    
    # Pre-extract IDs (minimal memory: 2L strings)
    ligand_ids = [sample_lr["source"].iloc[l].lower() for l in range(L)]
    receptor_ids = [sample_lr["target"].iloc[l].lower() for l in range(L)]
    
    def get_expression_value(cell_id, gene_id):
        """
        Memory-efficient expression extraction.
        Returns single scalar value, not storing intermediate tensors.
        """
        if len(gene_id.split("_")) > 1:
            gene_list = gene_id.split("_")
            # Create tensor, compute, return scalar - no persistent storage
            expr = torch.tensor(subdata[cell_id, gene_list].X.astype(np.float32))
            result = torch.exp(torch.mean(torch.log(expr + toll_geom_mean), dim=1)).item()
            del expr  # Explicit cleanup
            return result
        else:
            # Direct scalar extraction - minimal memory
            return float(subdata[cell_id, gene_id].X.flatten()[0])
    
    # Distance matrix computation - Memory: N × N × 4 bytes ≈ 4 MB for N=1000
    # This is unavoidable for vectorization but much smaller than main tensor
    coords_x = torch.tensor([subdata.obs.loc[cell_id, "centroid_x"] for cell_id in cell_indexes], 
                           dtype=torch.float32)
    coords_y = torch.tensor([subdata.obs.loc[cell_id, "centroid_y"] for cell_id in cell_indexes], 
                           dtype=torch.float32)
    
    # Compute distance matrix using broadcasting - Memory: N × N
    dist_matrix = torch.sqrt((coords_x.unsqueeze(1) - coords_x.unsqueeze(0))**2 + 
                           (coords_y.unsqueeze(1) - coords_y.unsqueeze(0))**2) + toll_dist
    
    # Clean up coordinate vectors
    del coords_x, coords_y
    
    # INTRALAYER INTERACTIONS: same layer (α == β), different cells (i ≠ j)
    # Process one layer at a time to minimize memory usage
    for alpha in range(L):
        ligand_id = ligand_ids[alpha]
        receptor_id = receptor_ids[alpha]
        
        # Extract expression vectors for this layer only - Memory: 2N scalars ≈ 8KB for N=1000
        ligand_vals = torch.tensor([get_expression_value(cell_indexes[i], ligand_id) 
                                  for i in range(N)], dtype=torch.float32)
        receptor_vals = torch.tensor([get_expression_value(cell_indexes[j], receptor_id) 
                                    for j in range(N)], dtype=torch.float32)
        
        # Create interaction matrix using broadcasting - Memory: N × N ≈ 4MB for N=1000
        # ligand_vals: (N,) -> (N, 1), receptor_vals: (N,) -> (1, N)
        # Broadcasting: (N, 1) × (1, N) → (N, N)
        interaction_matrix = torch.outer(ligand_vals, receptor_vals) / dist_matrix
        
        # Apply mask to exclude diagonal (same cell interactions)
        # More memory efficient than creating separate mask tensor
        interaction_matrix.fill_diagonal_(0)  # In-place operation
        
        # Assign to main tensor - direct assignment, no extra copies
        mlnet[:, alpha, :, alpha] = interaction_matrix
        
        # Clean up temporary arrays
        del ligand_vals, receptor_vals, interaction_matrix
    
    # INTERLAYER INTERACTIONS: different layers (α ≠ β), same cells (i == j)
    # Process in chunks to minimize memory usage
    for alpha in range(L):
        receptor_alpha_id = receptor_ids[alpha]
        
        # Extract receptor expression for all cells for this alpha
        receptor_alpha_vals = torch.tensor([get_expression_value(cell_indexes[i], receptor_alpha_id) 
                                          for i in range(N)], dtype=torch.float32)
        
        for beta in range(L):
            if alpha != beta:
                ligand_beta_id = ligand_ids[beta]
                
                # Extract ligand expression for all cells for this beta
                ligand_beta_vals = torch.tensor([get_expression_value(cell_indexes[i], ligand_beta_id) 
                                               for i in range(N)], dtype=torch.float32)
                
                # Element-wise multiplication for diagonal elements only
                # Memory: N scalars, not N×N matrix
                diagonal_values = receptor_alpha_vals * ligand_beta_vals
                
                # Assign to diagonal of main tensor using advanced indexing
                cell_indices = torch.arange(N)
                mlnet[cell_indices, alpha, cell_indices, beta] = diagonal_values
                
                # Clean up
                del ligand_beta_vals, diagonal_values
        
        # Clean up receptor values for this alpha
        del receptor_alpha_vals
    
    # Clean up distance matrix
    del dist_matrix
    
    toc = time.time()
    print(f"Memory-efficient vectorized time: {toc-tic:.2f} s")
    
    return mlnet

def build_multilayer_network_fully_vectorized_v3(N, L, cell_indexes, sample_lr, subdata, 
                                            toll_dist=1e-10, toll_geom_mean=1e-10,
                                            compute_intralayer=True, compute_interlayer=True,
                                            n_jobs=1):
    """
    Memory-efficient fully vectorized multilayer network construction.
    
    Args:
        N, L: Network dimensions
        cell_indexes, sample_lr, subdata: Data inputs
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
    
    tic = time.time()
    
    # Determine number of threads for parallelization
    if n_jobs == -1:
        import os
        n_jobs = os.cpu_count()
    elif n_jobs <= 0:
        n_jobs = 1
    
    print(f"Computing with {n_jobs} thread(s)")
    print(f"Intralayer: {'Yes' if compute_intralayer else 'No'}")
    print(f"Interlayer: {'Yes' if compute_interlayer else 'No'}")
    
    # Main tensor - this is our largest memory allocation
    mlnet = torch.zeros(N, L, N, L, dtype=torch.float32)
    
    # Pre-extract IDs (minimal memory: 2L strings)
    ligand_ids = [sample_lr["source"].iloc[l].lower() for l in range(L)]
    receptor_ids = [sample_lr["target"].iloc[l].lower() for l in range(L)]
    
    def get_expression_value(cell_id, gene_id):
        """Memory-efficient expression extraction."""
        if len(gene_id.split("_")) > 1:
            gene_list = gene_id.split("_")
            expr = torch.tensor(subdata[cell_id, gene_list].X.astype(np.float32))
            result = torch.exp(torch.mean(torch.log(expr + toll_geom_mean), dim=1)).item()
            del expr
            return result
        else:
            return float(subdata[cell_id, gene_id].X.flatten()[0])
    
    # Distance matrix computation (only if computing intralayer)
    dist_matrix = None
    if compute_intralayer:
        coords_x = torch.tensor([subdata.obs.loc[cell_id, "centroid_x"] for cell_id in cell_indexes], 
                               dtype=torch.float32)
        coords_y = torch.tensor([subdata.obs.loc[cell_id, "centroid_y"] for cell_id in cell_indexes], 
                               dtype=torch.float32)
        
        dist_matrix = torch.sqrt((coords_x.unsqueeze(1) - coords_x.unsqueeze(0))**2 + 
                               (coords_y.unsqueeze(1) - coords_y.unsqueeze(0))**2) + toll_dist
        del coords_x, coords_y
    
    # INTRALAYER INTERACTIONS: same layer (α == β), different cells (i ≠ j)
    if compute_intralayer:
        print("Computing intralayer interactions...")
        
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
    
    # INTERLAYER INTERACTIONS: different layers (α ≠ β), same cells (i == j)
    if compute_interlayer:
        print("Computing interlayer interactions...")
        
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
        
        # Generate all layer pairs (alpha != beta)
        layer_pairs = [(alpha, beta) for alpha in range(L) for beta in range(L) if alpha != beta]
        
        if n_jobs == 1:
            # Sequential processing
            for alpha, beta in layer_pairs:
                _, diagonal_values = compute_interlayer_for_pair((alpha, beta))
                cell_indices = torch.arange(N)
                mlnet[cell_indices, alpha, cell_indices, beta] = diagonal_values
                del diagonal_values
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
    
    # Clean up distance matrix if it was created
    if dist_matrix is not None:
        del dist_matrix
    
    toc = time.time()
    print(f"Memory-efficient vectorized time: {toc-tic:.2f} s")
    
    return mlnet


# Convenience functions for specific use cases
def build_intralayer_only(N, L, cell_indexes, sample_lr, subdata, 
                         toll_dist=1e-10, toll_geom_mean=1e-10, n_jobs=1):
    """Build only intralayer interactions (saves memory and computation)."""
    return build_multilayer_network_fully_vectorized(
        N, L, cell_indexes, sample_lr, subdata, toll_dist, toll_geom_mean,
        compute_intralayer=True, compute_interlayer=False, n_jobs=n_jobs)

def build_interlayer_only(N, L, cell_indexes, sample_lr, subdata, 
                         toll_dist=1e-10, toll_geom_mean=1e-10, n_jobs=1):
    """Build only interlayer interactions (saves memory and computation)."""
    return build_multilayer_network_fully_vectorized(
        N, L, cell_indexes, sample_lr, subdata, toll_dist, toll_geom_mean,
        compute_intralayer=False, compute_interlayer=True, n_jobs=n_jobs)

def calculate_memory_requirements(N, L):
    """
    Calculate expected memory usage for multilayer network tensor.
    
    Args:
        N: Number of nodes
        L: Number of layers
    
    Returns:
        Memory requirements in bytes and human-readable format
    """
    
    # Main tensor: N × L × N × L elements
    total_elements = N * L * N * L  # This is N² × L²
    bytes_per_element = 4  # float32
    main_tensor_bytes = total_elements * bytes_per_element
    
    # Auxiliary memory (much smaller):
    # - Distance matrix: N × N × 4 bytes
    # - Temporary vectors: ~2N × 4 bytes per layer
    # - Coordinate vectors: 2N × 4 bytes
    auxiliary_bytes = N * N * 4 + 2 * N * 4 + 2 * N * 4
    
    total_bytes = main_tensor_bytes + auxiliary_bytes
    
    # Convert to human-readable format
    def bytes_to_human(bytes_val):
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_val < 1024:
                return f"{bytes_val:.2f} {unit}"
            bytes_val /= 1024
        return f"{bytes_val:.2f} PB"
    
    print(f"Memory Analysis for N={N}, L={L}:")
    print(f"  Total elements in tensor: {total_elements:,}")
    print(f"  Main tensor memory: {bytes_to_human(main_tensor_bytes)}")
    print(f"  Auxiliary memory: {bytes_to_human(auxiliary_bytes)}")
    print(f"  Total expected memory: {bytes_to_human(total_bytes)}")
    print(f"  Main tensor dominates: {100 * main_tensor_bytes / total_bytes:.1f}% of total")
    
    return total_bytes

# Usage example:
# mlnet = build_multilayer_network_vectorized(N, L, cell_indexes, sample_lr, subdata)
# or for maximum speed (if you have enough memory):
#
x_hat_s = ad.read_h5ad(Path(__file__).parents[1] / "data" / "processed" / "mouse1_slice153_x_hat_s.h5ad")
print(x_hat_s)

source, target = "Astro", "L2/3 IT"
subdata = x_hat_s[x_hat_s.obs["subclass"].isin([source,target]), :]
print(subdata)

N = 500
L = 10

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

#mlnet = build_multilayer_network_vectorized(N, L, cell_indexes, sample_lr, subdata)
#mlnet_fully_vectorized = build_multilayer_network_fully_vectorized(N, L, cell_indexes, sample_lr, subdata)
#torch.save(mlnet, Path(__file__).parents[1] / "data" / "processed" / "mouse1_slice153_mlnet_broadcast.pt")
#torch.save(mlnet_fully_vectorized, Path(__file__).parents[1] / "data" / "processed" / "mouse1_slice153_mlnet_fully_vectorized.pt")
calculate_memory_requirements(N, L)
mlnet = build_multilayer_network_fully_vectorized_v2(N, L, cell_indexes, sample_lr, subdata)
torch.save(mlnet, Path(__file__).parents[1] / "data" / "processed" / "mouse1_slice153_mlnet_fully_vectorized_v2.pt")
mlnet = build_multilayer_network_fully_vectorized_v3(N, L, cell_indexes, sample_lr, subdata, n_jobs=2)
torch.save(mlnet, Path(__file__).parents[1] / "data" / "processed" / "mouse1_slice153_mlnet_fully_vectorized_v3.pt")
