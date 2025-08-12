import numpy as np
import torch
import pymnet
from mlnetst.utils.mlnet_utils import get_aggregate_from_supra_adjacency_matrix, build_supra_adjacency_matrix_from_tensor, binarize_matrix
import scipy as sp


def get_diagonal_blocks(n: int, l: int, device: torch.device = None) -> torch.Tensor:
    """
    Create a mask for diagonal blocks in a supra-adjacency matrix.

    Args:
        n (int): Number of nodes.
        l (int): Number of layers.
        device (torch.device, optional): Device to create the tensor on. Defaults to None.

    Returns:
        torch.Tensor: A boolean mask tensor of shape (N * L, N * L) with True values in the diagonal blocks.
    """

    # Create basic NxN block of ones
    block = torch.ones(n, n, dtype=torch.bool, device=device)
    
    # Create initial empty mask
    mask = torch.zeros(n*l, n*l, dtype=torch.bool, device=device)
    
    # Fill diagonal blocks
    for i in range(l):
        start_idx = i * n
        end_idx = (i + 1) * n
        mask[start_idx:end_idx, start_idx:end_idx] = block
        
    return mask

def get_non_diagonal_blocks(n: int, l: int, device: torch.device = None) -> torch.Tensor:
    """
    Create a mask for non-diagonal blocks in a supra-adjacency matrix.

    For a matrix of size (N*L x N*L) composed of LxL blocks of size (NxN),
    creates a boolean mask that keeps only the non-diagonal blocks.

    Args:
        n (int): Number of nodes.
        l (int): Number of layers.
        device (torch.device, optional): Device to create the tensor on. Defaults to None.

    Returns:
        torch.Tensor: A boolean mask tensor of shape (N*L, N*L) with True values 
                     in the non-diagonal blocks.
    
    Example for N=2, L=2:
        [[0 0 1 1]
         [0 0 1 1]
         [1 1 0 0]
         [1 1 0 0]]
    """
    # Create full mask of ones
    mask = torch.ones(n*l, n*l, dtype=torch.bool, device=device)
    
    # Create basic NxN block of ones for diagonal
    block = torch.ones(n, n, dtype=torch.bool, device=device)
    
    # Zero out diagonal blocks
    for i in range(l):
        start_idx = i * n
        end_idx = (i + 1) * n
        mask[start_idx:end_idx, start_idx:end_idx] = ~block
        
    return mask

def is_in_diagonal_block(indices: torch.Tensor, n: int) -> torch.Tensor:
    """
    Check which indices belong to diagonal blocks of a supra adjacency matrix.

    Args:
        indices: Tensor of shape (2, nnz) containing [row, col] indices
        n: Number of nodes per layer
        
    Returns:
        Boolean mask of length nnz indicating which indices are in diagonal blocks
    """
    # Get layer indices for rows and columns
    layer_row = indices[0] // n
    layer_col = indices[1] // n
    
    # Indices are in diagonal blocks if they're in the same layer
    return layer_row == layer_col

def compute_aggregated_indegree(supra_adjacency_matrix: torch.Tensor, n: int, l: int) -> torch.Tensor:
    """
    Compute aggregated indegree across all layers for each node.
    
    This function aggregates incoming edges from all layers for each node, considering only
    intra-layer connections (diagonal blocks of the supra-adjacency matrix).
    
    Args:
        supra_adjacency_matrix (torch.Tensor): Sparse tensor of shape (N*L, N*L) representing
            the supra-adjacency matrix where N is number of nodes and L is number of layers
        n (int): Number of nodes per layer
        l (int): Number of layers
        
    Returns:
        torch.Tensor: Tensor of shape (N,) containing aggregated indegrees for each node
        
    Note:
        This is more efficient than compute_indegree when you only need aggregated values
        across layers rather than per-layer breakdown.
    """
    # Get sparse indices and values
    supra_adjacency_matrix = supra_adjacency_matrix.coalesce()
    indices = supra_adjacency_matrix.indices()
    values = supra_adjacency_matrix.values()
    
    # find which indices are in diagonal blocks. One index is a pair of (row, col)
    diagonal_mask = is_in_diagonal_block(indices, n)
    
    # Filter indices and values
    diagonal_indices = indices[:, diagonal_mask]
    diagonal_values = values[diagonal_mask]
    
    # Create binary values for degree calculation
    binary_values = torch.ones_like(diagonal_values)
    
    # In-edges are rowsum, i take the column indices of the diagonal blocks and compute their node 
    # indices within layers. 
    node_indices = diagonal_indices[1] % n
    
    # Accumulate degrees
    nodes_indegrees = torch.zeros(n, device=supra_adjacency_matrix.device)
    nodes_indegrees.index_add_(0, node_indices, binary_values)
    
    return nodes_indegrees

def compute_indegree(supra_adjacency_matrix: torch.Tensor, n: int, l: int) -> torch.Tensor:
    """
    Compute indegree for each node replica in a multilayer network.
    
    This function computes the number of incoming edges for each node in each layer,
    treating each (node, layer) combination as a separate entity. Only considers
    intra-layer connections (diagonal blocks).
    
    Args:
        supra_adjacency_matrix (torch.Tensor): Sparse tensor of shape (N*L, N*L)
        n (int): Number of nodes per layer
        l (int): Number of layers
        
    Returns:
        torch.Tensor: Tensor of shape (N, L) where entry [i,j] is the indegree
            of node i in layer j
            
    Note:
        For aggregated indegree across layers, use compute_aggregated_indegree() 
        which is more efficient.
    """
    # Get sparse indices and values
    supra_adjacency_matrix = supra_adjacency_matrix.coalesce()
    indices = supra_adjacency_matrix.indices()
    values = supra_adjacency_matrix.values()

    # Find which indices are in diagonal blocks
    diagonal_mask = is_in_diagonal_block(indices, n)

    # Filter indices and values
    diagonal_indices = indices[:, diagonal_mask]
    diagonal_values = values[diagonal_mask]

    # Create binary values for degree calculation
    binary_values = torch.ones_like(diagonal_values)

    # In-edges are rowsum, i take the column indices of the diagonal blocks and compute their node 
    # indices within layers. Compute indegree for each layer simultaneously
    node_indegrees = torch.zeros((n, l), device=supra_adjacency_matrix.device)
    
    # Get target layer indices and node indices within layers
    target_layer_indices = diagonal_indices[1] // n  # Which layer the target node is in
    target_node_indices = diagonal_indices[1] % n    # Which node within that layer
    
    # Create combined indices for scatter_add: [node_idx * l + layer_idx]
    combined_indices = target_node_indices * l + target_layer_indices
    
    # Flatten the node_indegrees tensor and use scatter_add
    node_indegrees_flat = node_indegrees.view(-1)
    node_indegrees_flat.scatter_add_(0, combined_indices, binary_values)
    
    # Reshape back to (n, l)
    node_indegrees = node_indegrees_flat.view(n, l)
    print("Indegrees computed for each layer:", node_indegrees)
    return node_indegrees


def compute_indegree_for_layer(supra_adjacency_matrix: torch.Tensor, layer_index: int, n: int, l: int) -> torch.Tensor:
    """
    Compute indegree for a specific layer in a multilayer network.
    
    Args:
        supra_adjacency_matrix: Sparse tensor of shape (N*L, N*L)
        layer_index: Index of the layer to compute indegree for
        n: Number of nodes per layer
        l: Number of layers
        
    Returns:
        Tensor of shape (N,) containing indegrees for the specified layer
    """
    print(f"Computing indegree for layer {layer_index} with n={n}, l={l}")
    if layer_index < 0 or layer_index >= l:
        raise ValueError(f"Layer index {layer_index} is out of bounds for {l} layers.")
    
    # Calculate the start and end indices for the layer
    start_index = layer_index * n
    end_index = start_index + n

    if supra_adjacency_matrix.is_sparse:
        # Get sparse indices and values
        supra_adjacency_matrix = supra_adjacency_matrix.coalesce()
        indices = supra_adjacency_matrix.indices()
        values = supra_adjacency_matrix.values()
    else:
        raise NotImplementedError("Supra-adjacency matrix must be a sparse tensor.")
    
    # Create a mask for the specified layer
    layer_mask = (indices[0] >= start_index) & (indices[0] < end_index) & (indices[1] >= start_index) & (indices[1] < end_index)
    
    # Filter indices and values for the specified layer
    layer_indices = indices[:, layer_mask]
    layer_values = values[layer_mask]

    # Create binary values for degree calculation
    binary_values = torch.ones_like(layer_values)
    print(binary_values)

    node_indegrees = torch.zeros(n, device=supra_adjacency_matrix.device)
    # In-edges are rowsum, i take the column indices of the diagonal blocks and compute their node 
    # indices within layers. 
    node_indices = layer_indices[1] % n
    
    # Accumulate degrees
    node_indegrees.index_add_(0, node_indices, binary_values)
    return node_indegrees

def compute_instrength(supra_adjacency_matrix: torch.Tensor, n: int, l: int) -> torch.Tensor:
    """
    Compute the instrength of each node, considering them as replica nodes in a multilayer network.
    
    Args:
        supra_adjacency_matrix: Sparse tensor of shape (N*L, N*L)
        n: Number of nodes per layer
        l: Number of layers
    Returns:
        Tensor of shape (N*L,) containing instengths
    """
    # Get sparse indices and values
    supra_adjacency_matrix = supra_adjacency_matrix.coalesce()
    indices = supra_adjacency_matrix.indices()
    values = supra_adjacency_matrix.values()
    
    # Find which indices are in diagonal blocks
    diagonal_mask = is_in_diagonal_block(indices, n)
    
    # Filter indices and values
    diagonal_indices = indices[:, diagonal_mask]
    diagonal_values = values[diagonal_mask]
    
    # Compute node indices within layers
    node_indices = diagonal_indices[1] % n
    
    # Accumulate strengths
    nodes_instrengths = torch.zeros(n, device=supra_adjacency_matrix.device)
    nodes_instrengths.index_add_(0, node_indices, diagonal_values)
    
    return nodes_instrengths
    

def compute_outdegree(supra_adjacency_matrix: torch.Tensor, n: int, l: int) -> torch.Tensor:
    """
    Compute the outdegree of each node, considering them as replica nodes in a multilayer network.

    Args:
        supra_adjacency_matrix (torch.Tensor): The supra-adjacency matrix of shape (N * L, N * L).
        n (int): Number of nodes.
        l (int): Number of layers.

    Returns:
        torch.Tensor: A tensor of shape (N * L) containing the outdegree of each node.
    """
    # Get sparse indices and values
    supra_adjacency_matrix = supra_adjacency_matrix.coalesce()
    indices = supra_adjacency_matrix.indices()
    values = supra_adjacency_matrix.values()
    
    # Find which indices are in diagonal blocks
    diagonal_mask = is_in_diagonal_block(indices, n)
    
    # Filter indices and values
    diagonal_indices = indices[:, diagonal_mask]
    diagonal_values = values[diagonal_mask]
    
    # Create binary values for degree calculation
    binary_values = torch.ones_like(diagonal_values)
    
    # Compute node indices within layers
    node_indices = diagonal_indices[0] % n
    
    # Accumulate degrees
    nodes_outdegrees = torch.zeros(n, device=supra_adjacency_matrix.device)
    nodes_outdegrees.index_add_(0, node_indices, binary_values)

    return nodes_outdegrees

def compute_outstrength(supra_adjacency_matrix: torch.Tensor, n: int, l: int) -> torch.Tensor:
    """
    Compute the outstrength of each node, considering them as replica nodes in a multilayer network.

    Args:
        supra_adjacency_matrix (torch.Tensor): The supra-adjacency matrix of shape (N * L, N * L).
        n (int): Number of nodes.
        l (int): Number of layers.

    Returns:
        torch.Tensor: A tensor of shape (N * L) containing the outstrength of each node.
    """
    # Get sparse indices and values
    supra_adjacency_matrix = supra_adjacency_matrix.coalesce()
    indices = supra_adjacency_matrix.indices()
    values = supra_adjacency_matrix.values()

    # Find which indices are in diagonal blocks
    diagonal_mask = is_in_diagonal_block(indices, n)

    # Filter indices and values
    diagonal_indices = indices[:, diagonal_mask]
    diagonal_values = values[diagonal_mask]

    # Compute node indices within layers
    node_indices = diagonal_indices[0] % n

    # Accumulate strengths
    nodes_outstrengths = torch.zeros(n, device=supra_adjacency_matrix.device)
    nodes_outstrengths.index_add_(0, node_indices, diagonal_values)

    return nodes_outstrengths

def compute_multi_indegree(supra_adjacency_matrix: torch.Tensor, n:int, l: int) -> torch.Tensor:
    """
    Memory-efficient computation of multi-indegree for non-diagonal blocks.
    
    Args:
        supra_adjacency_matrix: Sparse tensor of shape (N*L, N*L)
        n: Number of nodes per layer
        l: Number of layers
        
    Returns:
        Tensor of shape (N,) containing multi-indegrees
    """
    # Get sparse indices and values
    supra_adjacency_matrix = supra_adjacency_matrix.coalesce()
    indices = supra_adjacency_matrix.indices()
    values = supra_adjacency_matrix.values()
    
    # Find which indices are in non-diagonal blocks
    non_diagonal_mask = ~is_in_diagonal_block(indices, n)
    
    # Filter indices and values
    non_diagonal_indices = indices[:, non_diagonal_mask]
    non_diagonal_values = values[non_diagonal_mask]
    
    # Create binary values for degree calculation
    binary_values = torch.ones_like(non_diagonal_values)
    
    # Compute node indices within layers
    node_indices = non_diagonal_indices[1] % n
    
    # Accumulate degrees
    nodes_multi_indegrees = torch.zeros(n, device=supra_adjacency_matrix.device)
    nodes_multi_indegrees.index_add_(0, node_indices, binary_values)
    
    return nodes_multi_indegrees

def compute_multi_instrength(supra_adjacency_matrix: torch.Tensor, n: int, l: int) -> torch.Tensor:
    """
    Memory-efficient computation of multi-instrength for non-diagonal blocks.
    
    Args:
        supra_adjacency_matrix: Sparse tensor of shape (N*L, N*L)
        n: Number of nodes per layer
        l: Number of layers
        
    Returns:
        Tensor of shape (N*L,) containing multi-instrengths
    """
    # Get sparse indices and values
    supra_adjacency_matrix = supra_adjacency_matrix.coalesce()
    indices = supra_adjacency_matrix.indices()
    values = supra_adjacency_matrix.values()
    
    # Find which indices are in non-diagonal blocks
    non_diagonal_mask = ~is_in_diagonal_block(indices, n)
    
    # Filter indices and values
    non_diagonal_indices = indices[:, non_diagonal_mask]
    
    #TODO #diagonal_indices = indices[:, ~non_diagonal_mask]
    non_diagonal_values = values[non_diagonal_mask]
    
    # Compute node indices within layers
    node_indices = non_diagonal_indices[1] % n
    
    # Accumulate strengths
    nodes_multi_instrengths = torch.zeros(n, device=supra_adjacency_matrix.device)
    nodes_multi_instrengths.index_add_(0, node_indices, non_diagonal_values)
    
    return nodes_multi_instrengths
    
def compute_multi_outdegree(supra_adjacency_matrix, n, l):
    """
    Memory-efficient computation of multi-outdegree for non-diagonal blocks.

    Args:
        supra_adjacency_matrix (torch.Tensor): Sparse tensor of shape (N * L, N * L).
        n (int): Number of nodes.
        l (int): Number of layers.
        
    Returns:
        torch.Tensor: A tensor of shape (N,) containing multi-outdegrees.
    """
    # Get sparse indices and values
    supra_adjacency_matrix = supra_adjacency_matrix.coalesce()
    indices = supra_adjacency_matrix.indices()
    values = supra_adjacency_matrix.values()
    
    # Find which indices are in non-diagonal blocks
    non_diagonal_mask = ~is_in_diagonal_block(indices, n)
    
    # Filter indices and values
    non_diagonal_indices = indices[:, non_diagonal_mask]
    non_diagonal_values = values[non_diagonal_mask]
    
    # Create binary values for degree calculation
    binary_values = torch.ones_like(non_diagonal_values)
    
    # Compute node indices within layers
    node_indices = non_diagonal_indices[0] % n
    
    # Accumulate degrees
    nodes_multi_outdegrees = torch.zeros(n, device=supra_adjacency_matrix.device)
    nodes_multi_outdegrees.index_add_(0, node_indices, binary_values)
    
    return nodes_multi_outdegrees

def compute_multi_outstrength(supra_adjacency_matrix, n, l):
    """
    Memory-efficient computation of multi-outstrength for non-diagonal blocks.

    Args:
        supra_adjacency_matrix (torch.Tensor): Sparse tensor of shape (N * L, N * L).
        n (int): Number of nodes.
        l (int): Number of layers.
        
    Returns:
        torch.Tensor: A tensor of shape (N * L,) containing multi-outstrengths.
    """
    # Get sparse indices and values
    supra_adjacency_matrix = supra_adjacency_matrix.coalesce()
    indices = supra_adjacency_matrix.indices()
    values = supra_adjacency_matrix.values()
    
    # Find which indices are in non-diagonal blocks
    non_diagonal_mask = ~is_in_diagonal_block(indices, n)
    
    # Filter indices and values
    non_diagonal_indices = indices[:, non_diagonal_mask]
    non_diagonal_values = values[non_diagonal_mask]
    
    # Compute node indices within layers
    node_indices = non_diagonal_indices[0] % n
    
    # Accumulate strengths
    nodes_multi_outstrengths = torch.zeros(n, device=supra_adjacency_matrix.device)
    nodes_multi_outstrengths.index_add_(0, node_indices, non_diagonal_values)
    
    return nodes_multi_outstrengths

def compute_all_basic_metrics(supra_adjacency_matrix: torch.Tensor, n: int, l: int) -> dict:
    """
    Efficiently compute all basic metrics in a single pass through the sparse matrix.
    
    This function is more efficient than calling individual metric functions when you need
    multiple metrics, as it processes the sparse matrix only once.
    
    Args:
        supra_adjacency_matrix (torch.Tensor): Sparse tensor of shape (N*L, N*L)
        n (int): Number of nodes per layer
        l (int): Number of layers
        
    Returns:
        dict: Dictionary containing all computed metrics:
            - 'aggregated_indegree': shape (N,)
            - 'aggregated_outdegree': shape (N,) 
            - 'aggregated_instrength': shape (N,)
            - 'aggregated_outstrength': shape (N,)
            - 'multi_indegree': shape (N,)
            - 'multi_outdegree': shape (N,)
            - 'multi_instrength': shape (N,)
            - 'multi_outstrength': shape (N,)
            - 'indegree_per_layer': shape (N, L)
    """
    # Single preprocessing step
    if not supra_adjacency_matrix.is_sparse:
        raise ValueError("Matrix must be sparse for efficient computation")
    
    supra_adjacency_matrix = supra_adjacency_matrix.coalesce()
    indices = supra_adjacency_matrix.indices()
    values = supra_adjacency_matrix.values()
    
    diagonal_mask = is_in_diagonal_block(indices, n)
    non_diagonal_mask = ~diagonal_mask
    
    # Split indices and values once
    diag_indices = indices[:, diagonal_mask]
    diag_values = values[diagonal_mask]
    non_diag_indices = indices[:, non_diagonal_mask]
    non_diag_values = values[non_diagonal_mask]
    
    device = supra_adjacency_matrix.device
    
    # Initialize result tensors
    results = {
        'aggregated_indegree': torch.zeros(n, device=device),
        'aggregated_outdegree': torch.zeros(n, device=device),
        'aggregated_instrength': torch.zeros(n, device=device),
        'aggregated_outstrength': torch.zeros(n, device=device),
        'multi_indegree': torch.zeros(n, device=device),
        'multi_outdegree': torch.zeros(n, device=device),
        'multi_instrength': torch.zeros(n, device=device),
        'multi_outstrength': torch.zeros(n, device=device),
        'indegree_per_layer': torch.zeros((n, l), device=device)
    }
    
    # Process diagonal blocks (intra-layer)
    if diag_indices.numel() > 0:
        diag_binary = torch.ones_like(diag_values)
        
        # Indegree and instrength (target nodes)
        target_nodes = diag_indices[1] % n
        results['aggregated_indegree'].index_add_(0, target_nodes, diag_binary)
        results['aggregated_instrength'].index_add_(0, target_nodes, diag_values)
        
        # Outdegree and outstrength (source nodes)
        source_nodes = diag_indices[0] % n
        results['aggregated_outdegree'].index_add_(0, source_nodes, diag_binary)
        results['aggregated_outstrength'].index_add_(0, source_nodes, diag_values)
        
        # Per-layer indegree
        target_layers = diag_indices[1] // n
        target_nodes_per_layer = target_nodes * l + target_layers
        results['indegree_per_layer'].view(-1).scatter_add_(0, target_nodes_per_layer, diag_binary)
    
    # Process non-diagonal blocks (inter-layer)
    if non_diag_indices.numel() > 0:
        non_diag_binary = torch.ones_like(non_diag_values)
        
        # Multi-indegree and multi-instrength
        multi_target_nodes = non_diag_indices[1] % n
        results['multi_indegree'].index_add_(0, multi_target_nodes, non_diag_binary)
        results['multi_instrength'].index_add_(0, multi_target_nodes, non_diag_values)
        
        # Multi-outdegree and multi-outstrength
        multi_source_nodes = non_diag_indices[0] % n
        results['multi_outdegree'].index_add_(0, multi_source_nodes, non_diag_binary)
        results['multi_outstrength'].index_add_(0, multi_source_nodes, non_diag_values)
    
    return results

def compute_total_degree(supra_adjacency_matrix: torch.Tensor, n: int, l: int) -> torch.Tensor:
    """
    Compute the total degree of each node across all layers.
    
    The total degree is the sum of intra-layer and inter-layer degrees (both in and out).
    This function uses the efficient batch computation method.
    
    Args:
        supra_adjacency_matrix (torch.Tensor): The supra-adjacency matrix of shape (N * L, N * L).
        n (int): Number of nodes.
        l (int): Number of layers.

    Returns:
        torch.Tensor: A tensor of shape (N,) containing the total degree of each node.
    """
    metrics = compute_all_basic_metrics(supra_adjacency_matrix, n, l)
    return (metrics['aggregated_indegree'] + metrics['aggregated_outdegree'] + 
            metrics['multi_indegree'] + metrics['multi_outdegree'])

def find_node_neighbors_within_layer(supra_adjacency_matrix: torch.Tensor, node_index: int, layer_index: int, n: int, l:int) -> torch.Tensor:
    """
    This function finds which are the nodes that are neighbors of a given node in a specific layer of a multilayer network.

    Args:
        supra_adjacency_matrix (torch.Tensor): The supra-adjacency matrix of shape (N * L, N * L).
        node_index (int): The index of the node for which to find neighbors.
        layer_index (int): The index of the layer in which to find neighbors.
        n (int): Number of nodes per layer.
        l (int): Number of layers.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tensor containing the local node indices within the layer and the unique neighbor node indices.
        Basically, it return the local node indices (replica identities) within the layer and the global node indices (across all layers in the supramatrix) of the neighbors.
    """


    if layer_index < 0 or layer_index >= l:
        raise ValueError(f"Layer index {layer_index} is out of bounds for {l} layers.")
    if node_index < 0 or node_index >= n:
        raise ValueError(f"Node index {node_index} is out of bounds for {n} nodes.")
    
    # Calculate the start and end indices for the layer
    start_index = layer_index * n
    end_index = start_index + n

    # find indices and values in the supra-adjacency matrix
    supra_adjacency_matrix = supra_adjacency_matrix.coalesce()
    indices = supra_adjacency_matrix.indices()

    # Create a mask of the indices that belong to the specified layer
    layer_mask = (indices[0] >= start_index) & (indices[0] < end_index) & (indices[1] >= start_index) & (indices[1] < end_index)

    # Filter indices and values for the specified layer
    layer_indices = indices[:, layer_mask]

    # Find neighbors of the specified node in the specified layer
    node_mask = (layer_indices[0] == (start_index + node_index)) | (layer_indices[1] == (start_index + node_index))
    neighbors_indices = layer_indices[:, node_mask]

    # Extract the unique neighbor node indices
    unique_neighbors = torch.unique(neighbors_indices[0].tolist() + neighbors_indices[1].tolist())
    
    # Convert to local node indices within the layer
    local_within_neighbors = unique_neighbors % n

    return local_within_neighbors, unique_neighbors

def find_node_neighbors_outside_layer(supra_adjacency_matrix: torch.Tensor, node_index: int, layer_index: int, n: int, l:int) -> torch.Tensor:
    """
    This function finds which are the nodes that are neighbors of a given node in all layers except the specified one.

    Args:
        supra_adjacency_matrix (torch.Tensor): The supra-adjacency matrix of shape (N * L, N * L).
        node_index (int): The index of the node for which to find neighbors.
        layer_index (int): The index of the layer in which to find neighbors.
        n (int): Number of nodes per layer.
        l (int): Number of layers.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tensor containing the local node indices within the layer and the unique neighbor node indices.
        Basically, it return the local node indices (replica identities) within the layer and the global node indices (across all layers in the supramatrix) of the neighbors.
    """
    if layer_index < 0 or layer_index >= l:
        raise ValueError(f"Layer index {layer_index} is out of bounds for {l} layers.")
    if node_index < 0 or node_index >= n:
        raise ValueError(f"Node index {node_index} is out of bounds for {n} nodes.")
    
    # Calculate the start and end indices for the layer
    start_index = layer_index * n
    end_index = start_index + n
    
    # find indices and values in the supra-adjacency matrix
    supra_adjacency_matrix = supra_adjacency_matrix.coalesce()
    indices = supra_adjacency_matrix.indices()

    # Create a mask of the indices that do not belong to the specified layer
    non_layer_mask = (indices[0] < start_index) | (indices[0] >= end_index) | (indices[1] < start_index) | (indices[1] >= end_index)

    # Filter indices and values for the non-specified layer
    non_layer_indices = indices[:, non_layer_mask]

    # Find neighbors of the specified node in the non-specified layer
    node_mask = (non_layer_indices[0] == (start_index + node_index)) | (non_layer_indices[1] == (start_index + node_index))
    neighbors_indices = non_layer_indices[:, node_mask]
    
    # Extract the unique neighbor node indices
    unique_neighbors = torch.unique(neighbors_indices[0].tolist() + neighbors_indices[1].tolist())

    # Convert to local node indices within the layer
    local_outside_neighbors = unique_neighbors % n
    
    return local_outside_neighbors, unique_neighbors

"""
Clustering coefficients are useful measures of transitivity in networks.
The local clustering coefficient of a node is the number of existing edges among the set of its neighboring nodes
divided by the total number of possible connections between them.

Several different definitoins for local clustering coefficients have been developed
for weighted and undirected networks and for directed networks.

Given a local clustering coefficient one can calculate a differnet global clustering coefficient by averaging over all nodes.
Alternatively, one can calculate a global clustering coefficient as the total number of closed triplets of nodes
divided by the number of connected triplets.

There are different ways to compute the clustering coefficient for multilayer networks.
Here we use the method proposed by De Domenico et al. (2013), "Mathematical formulation of multilayer networks".


Here we use the method proposed by Cozzo et al. (2015), "Structure of triadic relations in multiplex networks". (TBD)
"""

def get_sparse_trace(matrix: torch.Tensor) -> float:
    """
    Compute the trace of a sparse matrix efficiently.
    
    Args:
        matrix (torch.Tensor): Sparse tensor of shape (N, N).
        
    Returns:
        float: The trace of the matrix.
    """
    if not matrix.is_sparse:
        return torch.trace(matrix).item()
    
    matrix = matrix.coalesce()
    indices = matrix.indices()
    values = matrix.values()
    # The trace is the sum of diagonal elements, which are where row index == column index
    diagonal_mask = indices[0] == indices[1]
    return values[diagonal_mask].sum()

def compute_average_global_clustering(supra_adjacency_matrix: torch.Tensor, n: int, l: int) -> float:
    """
    Compute the average global clustering coefficient for a multilayer network.
    
    This implementation matches the R version from the reference, using the formula:
    C = tr(A²*A) / (max(A) * tr(A*F*A))
    where F is the matrix with 1s everywhere except the diagonal
    
    Uses an efficient sparse implementation that avoids memory explosion by computing
    only the diagonal elements of matrix products.
    
    Args:
        supra_adjacency_matrix: Sparse tensor of shape (N*L, N*L)
        n: Number of nodes per layer
        l: Number of layers
        
    Returns:
        float: Global clustering coefficient
        
    Reference:
        De Domenico et al. (2013) "Mathematical formulation of multilayer networks"
        Physical Review X, 3(4), 041022.
    """
    supra_adjacency_matrix = supra_adjacency_matrix.coalesce()
    indexes = supra_adjacency_matrix.indices()
    values = supra_adjacency_matrix.values()

    supra_adjacency_matrix = sp.sparse.coo_matrix(
        (values.cpu().numpy(),
        (indexes[0].cpu().numpy(), indexes[1].cpu().numpy())),
        shape=(n*l, n*l)
    )
    print("Supra-adjacency matrix shape:", supra_adjacency_matrix.shape)
    print("Supra-adjacency matrix nnz:", supra_adjacency_matrix.nnz)

    # Compute numerator: tr(A^2 * A)
    numerator = (supra_adjacency_matrix @ supra_adjacency_matrix @ supra_adjacency_matrix).diagonal().sum()
    # Compute the F matrix: Ones() - Eye()
    ones_matrix = np.ones((n*l, n*l), dtype=np.float32)
    eye_matrix = sp.sparse.coo_matrix(np.eye(n*l, dtype=np.float32))
    f_matrix = ones_matrix - eye_matrix

    # Compute the denominator: tr(A * F * A)
    denominator = (supra_adjacency_matrix @ f_matrix @ supra_adjacency_matrix).diagonal().sum()

    return numerator / (max(supra_adjacency_matrix.data) * denominator) if denominator != 0 else 0.0


def compute_local_clustering_coefficient(supra_adjacency_matrix: torch.Tensor, node_index: int, layer_index: int, n: int, l: int) -> float:
    f_matrix = torch.ones(n*l, n*l, dtype=torch.float32, device=supra_adjacency_matrix.device) - torch.eye(n*l, dtype=torch.float32, device=supra_adjacency_matrix.device)
    #