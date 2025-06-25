import torch

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
