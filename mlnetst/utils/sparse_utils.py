import torch
import numpy as np
from typing import Tuple, List, Union, Dict

def create_layer_gene_mapping(ligand_ids: List[str], receptor_ids: List[str], var_names: List[str]) -> Dict[int, Dict[str, Dict[str, List[int]]]]:
    """
    Create a unified mapping between layer indices and both ligand and receptor components.
    
    Args:
        ligand_ids: List of ligand IDs (can include complex genes separated by '_')
        receptor_ids: List of receptor IDs (can include complex genes separated by '_')
        var_names: List of variable names available in the dataset
        
    Returns:
        Dictionary mapping layer index to both ligand and receptor information:
        {
            layer_idx: {
                "ligand": {
                    "gene_id": original ligand id,
                    "component_indices": list of indices in var_names for each component
                },
                "receptor": {
                    "gene_id": original receptor id,
                    "component_indices": list of indices in var_names for each component
                }
            }
        }
    """
    layer_map = {}
    
    for layer_idx, (ligand_id, receptor_id) in enumerate(zip(ligand_ids, receptor_ids)):
        layer_info = {"ligand": {}, "receptor": {}}
        valid_layer = True
        
        # Process ligand
        ligand_components = ligand_id.split("_")
        ligand_indices = []
        for component in ligand_components:
            try:
                idx = var_names.tolist().index(component)
                ligand_indices.append(idx)
            except ValueError:
                valid_layer = False
                break
                
        # Process receptor
        receptor_components = receptor_id.split("_")
        receptor_indices = []
        for component in receptor_components:
            try:
                idx = var_names.tolist().index(component)
                receptor_indices.append(idx)
            except ValueError:
                valid_layer = False
                break
        
        # Only add to mapping if all components were found
        if valid_layer and ligand_indices and receptor_indices:
            layer_info["ligand"] = {
                "gene_id": ligand_id,
                "component_indices": ligand_indices
            }
            layer_info["receptor"] = {
                "gene_id": receptor_id,
                "component_indices": receptor_indices
            }
            layer_map[layer_idx] = layer_info
            
    return layer_map

def select_intra_layers(ligand_ids: List[str], var_names: List[str]) -> List[int]:
    """
    Select indices of layers based on ligand IDs.
    
    Args:
        ligand_ids: List of ligand IDs to select
        var_names: List of variable names (layer names)
        
    Returns:
        List of indices corresponding to the selected layers
    """
    return [var_names.tolist().index(ligand_id) for ligand_id in ligand_ids if ligand_id in var_names]

def get_expression_value_batch(data, gene_indexes, toll_complex):
    if len(gene_indexes) > 1:  # gene_id is a complex
        expr = torch.tensor(data[:, gene_indexes].X.astype(np.float32),
                            dtype=torch.float32)
        return torch.exp(torch.mean(torch.log(expr + toll_complex), dim=1)).squeeze()
    else:
        return torch.tensor(data[:, gene_indexes[0]].X.astype(np.float32),
                            dtype=torch.float32).squeeze()

def compute_intralayer_interactions(data, dist_matrix, src_idx, layer_src_info: Dict[str, List[int]], toll_complex: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute intralayer interactions for a specific layer from a sparse tensor.
    
    Args:
        sparse_tensor: torch.sparse.FloatTensor
        alpha: Layer index to compute interactions for
        
    Returns:
        Tuple of lists containing cell indices and interaction values
    """
    ligand_name = layer_src_info["ligand"]["gene_id"]
    ligand_indices = layer_src_info["ligand"]["component_indices"]
    receptor_name = layer_src_info["receptor"]["gene_id"]
    receptor_indices = layer_src_info["receptor"]["component_indices"]
    ligand_vals = get_expression_value_batch(data, ligand_indices, toll_complex)
    receptor_vals = get_expression_value_batch(data, receptor_indices, toll_complex)
    # print(f"Shape of ligand values: {ligand_vals.shape}, receptor values: {receptor_vals.shape}")  # Debugging output
    # print(f"Shape of outer product: {torch.outer(ligand_vals, receptor_vals).shape}")  # Debugging output
    # print(f"Shape of distance matrix: {dist_matrix.shape}")  # Debugging output
    # print(f"How many inf values in distance matrix: {torch.isinf(dist_matrix).sum().item()}")  # Debugging output
    # compute interaction matrix as outer product of ligand and receptor values
    interaction_matrix = torch.outer(ligand_vals, receptor_vals) / dist_matrix
    # Check for NaN values in interaction matrix
    # print(f"Number of NaN values in interaction matrix: {torch.isnan(interaction_matrix).sum().item()}")  # Debugging output
    # print(f"Number of non-zero elements in interaction matrix before removing self-interactions: {interaction_matrix.nonzero().size(0)}")  # Debugging output
    # sanity check that there are no self-interactions
    interaction_matrix.fill_diagonal_(0)  # Remove self-interactions
    # print(f"Number of non-zero elements in interaction matrix {interaction_matrix.nonzero()} for layer {src_idx} with ligand {ligand_name} and receptor {receptor_name}")  # Debugging output
    # Get non-zero indices and values
    nonzero_positions = torch.nonzero(interaction_matrix, as_tuple=False)
    nonzero_values = interaction_matrix[nonzero_positions[:, 0], nonzero_positions[:, 1]]
    
    # convert to 4D indices: [i, alpha, j, alpha]
    layer_indices = []
    layer_values = []
    if len(nonzero_positions)>0:
        i_indices = nonzero_positions[:, 0]
        j_indices = nonzero_positions[:, 1]
        alpha_indices = torch.full_like(i_indices, src_idx)
        # Stack as [dim0, dim1, dim2, dime3] = [i, alpha, j, alpha]
        layer_indices = torch.stack([i_indices, alpha_indices, j_indices, alpha_indices])
        layer_values = nonzero_values
    del interaction_matrix, ligand_vals, receptor_vals, nonzero_positions, nonzero_values
    return layer_indices, layer_values

def compute_interlayer_interactions(data, dist_matrix, src_layer: int, dst_layer: int, src_info: Dict[str, List[int]], dst_info: Dict[str, List[int]], toll_complex: float) -> Tuple[torch.Tensor, torch.Tensor]:

    receptor_src_name = src_info["receptor"]["gene_id"]
    receptor_src_indices = src_info["receptor"]["component_indices"]
    ligand_dst_name = dst_info["ligand"]["gene_id"]
    ligand_dst_indices = dst_info["ligand"]["component_indices"]
    receptor_vals = get_expression_value_batch(data, receptor_src_indices, toll_complex)
    ligand_vals = get_expression_value_batch(data, ligand_dst_indices, toll_complex)

    diagonal_values = receptor_vals * ligand_vals
    nonzero_positions = torch.nonzero(diagonal_values, as_tuple=False).squeeze()
    if nonzero_positions.numel() == 0:
        return torch.empty((0, 4), dtype=torch.long), torch.empty(0, dtype=torch.float32)
    nonzero_values = diagonal_values[nonzero_positions]
    
    # convert to 4D indices: [i, alpha, i, beta]
    pair_indices = []
    pair_values = []
    if len(nonzero_positions) > 0:
        i_indices = nonzero_positions
        src_indexes = torch.full_like(i_indices, src_layer)
        dst_indexes = torch.full_like(i_indices, dst_layer)
        # Stack as [dim0, dim1, dim2, dime3] = [i, alpha, i, beta]
        pair_indices = torch.stack([i_indices, src_indexes, i_indices, dst_indexes])
        pair_values = nonzero_values
    del ligand_vals, receptor_vals, diagonal_values, nonzero_positions, nonzero_values
    return pair_indices, pair_values
    

def compute_distance_matrix(cell_indexes, coord_x, coord_y, toll_distance=1e-6) -> torch.FloatTensor:
    """
    Compute a distance matrix for cells based on their coordinates.
    
    Args:
        cell_indexes: List of cell indices
        coord_x: List of x-coordinates for each cell
        coord_y: List of y-coordinates for each cell
        toll_distance: Distance threshold to consider interactions

    Returns:
        torch.FloatTensor: Distance matrix with shape (N, N)
    """
    N = len(cell_indexes)

    dist_matrix = torch.sqrt(
        (torch.tensor(coord_x).view(-1, 1) - torch.tensor(coord_x).view(1, -1)) ** 2 +
        (torch.tensor(coord_y).view(-1, 1) - torch.tensor(coord_y).view(1, -1)) ** 2
    ) + toll_distance  # Add tolerance to avoid division by zero
    dist_matrix = dist_matrix.fill_diagonal_(torch.inf)  # Remove self-interactions by setting diagonal to 0
    return dist_matrix


def get_layer_interaction(sparse_tensor, alpha, beta=None):
    """
    Extract interactions for specific layer(s) from sparse tensor.
    
    Args:
        sparse_tensor: torch.sparse.FloatTensor
        alpha: Source layer index
        beta: Target layer index (if None, returns intralayer interactions for alpha)
        
    Returns:
        torch.FloatTensor: 2D interaction matrix for the specified layer(s)
    """
    if beta is None:
        beta = alpha
    
    # Get indices and values
    indices = sparse_tensor.indices()
    values = sparse_tensor.values()
    
    # Find entries where dim1 == alpha and dim3 == beta
    mask = (indices[1] == alpha) & (indices[3] == beta)
    
    if not mask.any():
        # No interactions for this layer pair
        N = sparse_tensor.size(0)
        return torch.zeros(N, N, dtype=torch.float32)
    
    # Extract relevant indices and values
    relevant_indices = indices[:, mask]
    relevant_values = values[mask]
    
    # Create 2D sparse tensor
    N = sparse_tensor.size(0)
    layer_sparse = torch.sparse_coo_tensor(
        indices=relevant_indices[[0, 2]],  # [i, j] indices
        values=relevant_values,
        size=(N, N),
        dtype=torch.float32
    )
    
    return layer_sparse.to_dense()


def get_cell_interactions(sparse_tensor, cell_idx):
    """
    Extract all interactions involving a specific cell from sparse tensor.
    
    Args:
        sparse_tensor: torch.sparse.FloatTensor
        cell_idx: Cell index to extract interactions for
        
    Returns:
        dict: Dictionary with keys 'outgoing' and 'incoming' containing interaction matrices
    """
    indices = sparse_tensor.indices()
    values = sparse_tensor.values()
    
    L = sparse_tensor.size(1)
    
    # Outgoing interactions: cell_idx as source (dim0 == cell_idx)
    outgoing_mask = indices[0] == cell_idx
    outgoing_indices = indices[:, outgoing_mask]
    outgoing_values = values[outgoing_mask]
    
    # Incoming interactions: cell_idx as target (dim2 == cell_idx)
    incoming_mask = indices[2] == cell_idx
    incoming_indices = indices[:, incoming_mask]
    incoming_values = values[incoming_mask]
    
    # Create sparse matrices for outgoing [source_layer, target_cell, target_layer]
    if outgoing_mask.any():
        outgoing_sparse = torch.sparse_coo_tensor(
            indices=torch.stack([outgoing_indices[1], outgoing_indices[2], outgoing_indices[3]]),
            values=outgoing_values,
            size=(L, sparse_tensor.size(2), L),
            dtype=torch.float32
        )
    else:
        outgoing_sparse = torch.sparse_coo_tensor(
            indices=torch.zeros((3, 0), dtype=torch.long),
            values=torch.zeros(0, dtype=torch.float32),
            size=(L, sparse_tensor.size(2), L),
            dtype=torch.float32
        )
    
    # Create sparse matrices for incoming [source_cell, source_layer, target_layer]
    if incoming_mask.any():
        incoming_sparse = torch.sparse_coo_tensor(
            indices=torch.stack([incoming_indices[0], incoming_indices[1], incoming_indices[3]]),
            values=incoming_values,
            size=(sparse_tensor.size(0), L, L),
            dtype=torch.float32
        )
    else:
        incoming_sparse = torch.sparse_coo_tensor(
            indices=torch.zeros((3, 0), dtype=torch.long),
            values=torch.zeros(0, dtype=torch.float32),
            size=(sparse_tensor.size(0), L, L),
            dtype=torch.float32
        )
    
    return {
        'outgoing': outgoing_sparse,
        'incoming': incoming_sparse
    }
