import numpy as np
import torch
import pymnet
from mlnetst.utils.mlnet_utils import get_aggregate_from_supra_adjacency_matrix, build_supra_adjacency_matrix_from_tensor, binarize_matrix
import scipy as sp

"""
Utilities for computing various metrics on multilayer networks.
"""
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

def get_eigenvalues(supra_adjacency_matrix: torch.Tensor, n: int, l: int) -> float:
    """
    Compute the largest eigenvalue of the supra-adjacency matrix.
    Uses the power iteration method for efficiency.
    Args:

        supra_adjacency_matrix (torch.Tensor): Sparse tensor of shape (N*L, N*L)
        n (int): Number of nodes per layer
        l (int): Number of layers
    Returns:

        float: The largest eigenvalue of the supra-adjacency matrix.

    """
    # Ensure the matrix is sparse
    if not supra_adjacency_matrix.is_sparse:
        raise ValueError("Supra-adjacency matrix must be a sparse tensor.")
    indices = supra_adjacency_matrix.indices()
    values = supra_adjacency_matrix.values()
    sparse_array = sp.sparse.coo_matrix(
        (values.cpu().numpy(), (indices[0].cpu().numpy(), indices[1].cpu().numpy())),
        shape=(n * l, n * l)
    )
    eigen = sp.sparse.linalg.eigs(sparse_array)
    return eigen
    
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


"""
Centrality measures are used to identify the most important nodes in a network.
"""
# Versatility

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

    # In-edges are rowsum, i take the column indices of the diagonal blocks and compute their node 
    # indices within layers. Compute indegree for each layer simultaneously
    node_indegrees = torch.zeros((n, l), device=supra_adjacency_matrix.device)
    
    # Get target layer indices and node indices within layers
    target_layer_indices = diagonal_indices[1] // n  # Which layer the target node is in
    target_node_indices = diagonal_indices[1] % n    # Which node within that layer
    
    # Create combined indices for scatter_add: [node_idx * l + layer_idx]
    combined_indices = target_node_indices * l + target_layer_indices
    
    # Flatten the node_indegrees tensor and use scatter_add
    node_instrengths_flat = node_indegrees.view(-1)
    node_instrengths_flat.scatter_add_(0, combined_indices, diagonal_values)
    
    # Reshape back to (n, l)
    node_instrengths = node_instrengths_flat.view(n, l)
    print("Instrengths computed for each layer:", node_instrengths)
    return node_instrengths

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

    # Find which indices
    diagonal_mask = is_in_diagonal_block(indices, n)

    # Filter indices and values
    diagonal_indices = indices[:, diagonal_mask]
    diagonal_values = values[diagonal_mask]

    # Create binary values for degree calculation
    binary_values = torch.ones_like(diagonal_values)

    # In-edges are rowsum, i take the column indices of the diagonal blocks and compute their node 
    # indices within layers. Compute indegree for each layer simultaneously
    node_outdegrees = torch.zeros((n, l), device=supra_adjacency_matrix.device)
    
    # Get target layer indices and node indices within layers
    target_layer_indices = diagonal_indices[0] // n  # Which layer the target node is in
    target_node_indices = diagonal_indices[0] % n    # Which node within that layer
    
    # Create combined indices for scatter_add: [node_idx * l + layer_idx]
    combined_indices = target_node_indices * l + target_layer_indices
    
    # Flatten the node_outdegrees tensor and use scatter_add
    node_outdegrees_flat = node_outdegrees.view(-1)
    node_outdegrees_flat.scatter_add_(0, combined_indices, binary_values)
    
    # Reshape back to (n, l)
    node_outdegrees = node_outdegrees_flat.view(n, l)
    print("Outdegrees computed for each layer:", node_outdegrees)
    return node_outdegrees

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

    # In-edges are rowsum, i take the column indices of the diagonal blocks and compute their node 
    # indices within layers. Compute indegree for each layer simultaneously
    node_outdegrees = torch.zeros((n, l), device=supra_adjacency_matrix.device)
    
    # Get target layer indices and node indices within layers
    target_layer_indices = diagonal_indices[0] // n  # Which layer the target node is in
    target_node_indices = diagonal_indices[0] % n    # Which node within that layer
    
    # Create combined indices for scatter_add: [node_idx * l + layer_idx]
    combined_indices = target_node_indices * l + target_layer_indices
    
    # Flatten the node_outdegrees tensor and use scatter_add
    node_outdegrees_flat = node_outdegrees.view(-1)
    node_outdegrees_flat.scatter_add_(0, combined_indices, diagonal_values)
    
    # Reshape back to (n, l)
    node_outdegrees = node_outdegrees_flat.view(n, l)
    print("Outdegrees computed for each layer:", node_outdegrees)
    return node_outdegrees

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

    # Find which indices are in diagonal blocks
    not_diagonal_mask = ~is_in_diagonal_block(indices, n)

    # Filter indices and values
    not_diagonal_indices = indices[:, not_diagonal_mask]
    not_diagonal_values = values[not_diagonal_mask]

    # Create binary values for degree calculation
    binary_values = torch.ones_like(not_diagonal_values)

    # In-edges are rowsum, i take the column indices of the diagonal blocks and compute their node 
    # indices within layers. Compute indegree for each layer simultaneously
    node_indegrees = torch.zeros((n, l), device=supra_adjacency_matrix.device)
    
    # Get target layer indices and node indices within layers
    target_layer_indices = not_diagonal_indices[1] // n  # Which layer the target node is in
    target_node_indices = not_diagonal_indices[1] % n    # Which node within that layer
    
    # Create combined indices for scatter_add: [node_idx * l + layer_idx]
    combined_indices = target_node_indices * l + target_layer_indices
    
    # Flatten the node_indegrees tensor and use scatter_add
    node_indegrees_flat = node_indegrees.view(-1)
    node_indegrees_flat.scatter_add_(0, combined_indices, binary_values)
    
    # Reshape back to (n, l)
    node_indegrees = node_indegrees_flat.view(n, l)
    print("Indegrees computed for each layer:", node_indegrees)
    return node_indegrees

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

    # Find which indices are in diagonal blocks
    not_diagonal_mask = ~is_in_diagonal_block(indices, n)

    # Filter indices and values
    not_diagonal_indices = indices[:, not_diagonal_mask]
    not_diagonal_values = values[not_diagonal_mask]

    # In-edges are rowsum, i take the column indices of the diagonal blocks and compute their node 
    # indices within layers. Compute indegree for each layer simultaneously
    node_indegrees = torch.zeros((n, l), device=supra_adjacency_matrix.device)
    
    # Get target layer indices and node indices within layers
    target_layer_indices = not_diagonal_indices[1] // n  # Which layer the target node is in
    target_node_indices = not_diagonal_indices[1] % n    # Which node within that layer
    
    # Create combined indices for scatter_add: [node_idx * l + layer_idx]
    combined_indices = target_node_indices * l + target_layer_indices
    
    # Flatten the node_indegrees tensor and use scatter_add
    node_indegrees_flat = node_indegrees.view(-1)
    node_indegrees_flat.scatter_add_(0, combined_indices, not_diagonal_values)
    
    # Reshape back to (n, l)
    node_indegrees = node_indegrees_flat.view(n, l)
    print("Indegrees computed for each layer:", node_indegrees)
    return node_indegrees
    
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

    # Find which indices are in diagonal blocks
    not_diagonal_mask = ~is_in_diagonal_block(indices, n)

    # Filter indices and values
    not_diagonal_indices = indices[:, not_diagonal_mask]
    not_diagonal_values = values[not_diagonal_mask]

    # Create binary values for degree calculation
    binary_values = torch.ones_like(not_diagonal_values)

    # In-edges are rowsum, i take the column indices of the diagonal blocks and compute their node 
    # indices within layers. Compute indegree for each layer simultaneously
    node_outdegrees = torch.zeros((n, l), device=supra_adjacency_matrix.device)
    
    # Get target layer indices and node indices within layers
    target_layer_indices = not_diagonal_indices[0] // n  # Which layer the target node is in
    target_node_indices = not_diagonal_indices[0] % n    # Which node within that layer
    
    # Create combined indices for scatter_add: [node_idx * l + layer_idx]
    combined_indices = target_node_indices * l + target_layer_indices
    
    # Flatten the node_indegrees tensor and use scatter_add
    node_outdegrees_flat = node_outdegrees.view(-1)
    node_outdegrees_flat.scatter_add_(0, combined_indices, binary_values)
    
    # Reshape back to (n, l)
    node_outdegrees = node_outdegrees_flat.view(n, l)
    print("Outdegrees computed for each layer:", node_outdegrees)
    return node_outdegrees

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

    # Find which indices are in diagonal blocks
    not_diagonal_mask = ~is_in_diagonal_block(indices, n)

    # Filter indices and values
    not_diagonal_indices = indices[:, not_diagonal_mask]
    not_diagonal_values = values[not_diagonal_mask]

    # In-edges are rowsum, i take the column indices of the diagonal blocks and compute their node 
    # indices within layers. Compute indegree for each layer simultaneously
    node_outstrengths = torch.zeros((n, l), device=supra_adjacency_matrix.device)
    
    # Get target layer indices and node indices within layers
    target_layer_indices = not_diagonal_indices[0] // n  # Which layer the target node is in
    target_node_indices = not_diagonal_indices[0] % n    # Which node within that layer

    # Create combined indices for scatter_add: [node_idx * l + layer_idx]
    combined_indices = target_node_indices * l + target_layer_indices

    # Flatten the node_outstrengths tensor and use scatter_add
    node_outstrengths_flat = node_outstrengths.view(-1)
    node_outstrengths_flat.scatter_add_(0, combined_indices, not_diagonal_values)

    # Reshape back to (n, l)
    node_outstrengths = node_outstrengths_flat.view(n, l)
    print("Outstrengths computed for each layer:", node_outstrengths)
    return node_outstrengths

# K-coreness
#TODO

# Eigenvector versatility
def compute_katz_centrality(supra_adjacency_matrix: torch.Tensor, n: int, l: int) -> torch.Tensor:
    """
    Compute the eigenvector versatility for each node in a multilayer network.
    From Manlio et al. "Multilayer network science"
    << Eigenvector centrality is a measure of influence of the single nodes in a network. 
    A particular node has a high eigenvector centrality score when its neighbors have high score themselves.
    In the monoplex case, the recursive character of this definitn is untangled by the eigenvalue problem.
    In the multilayer case, the eigenvector centrality is defined as the solution of the generalized eigenvalue problem.
        Wx = lambda1 x
    Where lambda1 is the largest eigenvalue of W and the element x represent the centrality score of the node i
    If W is positive and symmetric, then the Perron-Fobrenius theorem grants the existence and uniqueness of this vecotr.

    An extension to multilayer network is obtained as:

       \sum_{i,\alpha} M_{j\beta}^{i\\alpha} \\Sigma_{i\\alpha} = \lambda1\ \Sigma_{j\beta}

    Where \lambda1 is the largest eigenvalue of M and \Sigma is teh corresponding eigentensor, whose values represent the centrality of
    each node in each layer.
    Thus, the eigenvector centraily consists in:

       \Sigma_{j\beta} = \lambda_1^{-1} \sum_{i,\alpha} M_{j\beta}^{i\alpha} \Sigma_{i\alpha}

    Which represents the multilayer generalization of Bonacich's eigenvector centrality per node per layer. By summing
    up the scores of a node across ll the layers, eigenvector versatility can be condensed across layers: \sigma_i = \sum_{\alpha} \Sigma_{i\alpha}
    
    In case of directed netwokr we refer to the KATZ versatility. This overcome some issues when outgoing edges are not
    present for some nodes. Katz assigns to each node a minimum value of centrality redefining the previous as:

        \sum_{i\alpha} (aM_{j\beta}^{i\alpha}) \Sigma_{i\alpha} + 1) = \Sigma_{j\beta}

    >>
    """
    supra_adjacency_matrix = supra_adjacency_matrix.coalesce()
    indexes = supra_adjacency_matrix.indices()
    # Use binary values for connectivity (same behavior as before) but ensure float dtype for SciPy
    binary_values = torch.ones_like(supra_adjacency_matrix.values(), dtype=torch.float32)

    # Build sparse COO and convert to CSC for column-oriented operations / factorization
    coo = sp.sparse.coo_matrix(
        (binary_values.cpu().numpy().astype(np.float64),
         (indexes[0].cpu().numpy(), indexes[1].cpu().numpy())),
        shape=(n * l, n * l)
    )
    # Ensure we have a canonical CSC matrix (sorted indices, duplicates summed)
    supra_csc = coo.tocsc()
    try:
        supra_csc.sum_duplicates()
    except Exception:
        # older scipy versions may not have sum_duplicates on csc; ignore if not present
        pass
    supra_csc.sort_indices()
    supra_adjacency_matrix = supra_csc
    print("Converted supra-adjacency matrix to canonical CSC format.")

    # Compute leading eigenvalue (use k=1 to request only the largest eigenvalue)
    try:
        eigvals = sp.sparse.linalg.eigs(supra_adjacency_matrix.T, k=1, which='LM', return_eigenvectors=False)
        leading_eigenvalue = eigvals[0].real
    except Exception:
        # fallback: request a single eigenvalue without extras (may still fail on tiny matrices)
        eigvals = sp.sparse.linalg.eigs(supra_adjacency_matrix.T, k=1, return_eigenvectors=False)
        leading_eigenvalue = eigvals[0].real
    print("Leading eigenvalue:", leading_eigenvalue)

    # Build delta as canonical CSC and ensure float dtype
    delta = sp.sparse.kron(sp.sparse.eye(n, format="csc", dtype=np.float64),
                           sp.sparse.eye(l, format="csc", dtype=np.float64))
    print("Delta shape:", delta.shape)

    # This ensures convergence of the Katz kernel tensor
    if leading_eigenvalue == 0:
        a = 0.99999
    else:
        a = 0.99999 / abs(leading_eigenvalue)

    # Form the linear operator (delta - a * supra) and canonicalize to CSC
    A = delta - a * supra_adjacency_matrix
    A = A.tocsc()
    print("Linear system matrix (delta - a*M) is canonical CSC with shape:", A.shape)

    # Solve the linear system A x = 1 instead of computing a dense inverse
    ones = np.ones(n * l, dtype=np.float64)
    katz_vec = sp.sparse.linalg.spsolve(A, ones)
    print("Solved linear system for Katz centrality, vector length:", katz_vec.shape)

    # Reshape to (n, l) and aggregate across layers
    katz_centrality = katz_vec.reshape(n, l)
    katz_centrality = katz_centrality.sum(axis=1)

    # Normalize and return as torch tensor (float32)
    maxv = katz_centrality.max()
    if maxv == 0:
        centrality = np.zeros_like(katz_centrality, dtype=np.float32)
    else:
        centrality = (katz_centrality / maxv).astype(np.float32)

    return torch.from_numpy(centrality)

def compute_multipagerank_centrality(supra_adjacency_matrix: torch.Tensor, n: int, l: int, alpha: float = 0.85) -> torch.Tensor:
    """
    Compute the multipagerank centrality for each node in a multilayer network.
    
    Args:
        supra_adjacency_matrix (torch.Tensor): Sparse tensor of shape (N*L, N*L)
        n (int): Number of nodes per layer
        l (int): Number of layers
        alpha (float): Damping factor for PageRank, typically between 0 and 1
        
    Returns:
        torch.Tensor: Tensor of shape (N,) containing multipagerank centrality for each node
    """
    pass

def compute_betweenness_centrality(supra_adjacency_matrix: torch.Tensor, n: int, l: int) -> torch.Tensor:
    """
    Given the existence of lazers, it is possible to select a subset of lazers and consider onlz the paths 
    belonging to that subset, defining the cross between centrality of the node i as:
        
        CBC(i,\alpha,\Omega) = \sum_{o\neq i,d\neq i,\beta\in\Omega,\beta\neq\beta'} \frac{\sigma_{o\beta}^{d\beta'}(i,\alpha)}{\sigma\}
        
    Where CBC counts the fraction of interlayer shortest paths, having their destination in \Omega, that pass through node i of layer \alpha. 

    Therefore, the multiplex betweenness centrality (BC) can be decomposed into the following contributions:

        BC(i,\alpha) = CBC(i,\alpha,\Omega) + CBC(i,\alpha,\bar{\Omega}) + IBC(i,\alpha)

    Where \bar{\Omega} indicates the layers that do not belong to the subset \Omega, and IBC(i,\alpha) is the interlayer betweenness centrality of node i in layer \alpha, defined as:

        IBC(i,\alpha) = \sum_{o\neq i,d\neq i,\alpha=\beta=\beta'} \frac{\sigma_{o\beta}^{d\beta'}(i,\alpha)}{\sigma}
        
    Compute the betweenness centrality for each node in a multilayer network.
    
    Args:
        supra_adjacency_matrix (torch.Tensor): Sparse tensor of shape (N*L, N*L)
        n (int): Number of nodes per layer
        l (int): Number of layers
        
    Returns:
        torch.Tensor: Tensor of shape (N,) containing betweenness centrality for each node
    """


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
    from time import time
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

    # Convert supra_adjacency_matrix to CSR format for efficient multiplication
    supra_adjacency_matrix = supra_adjacency_matrix.tocsr()
    print("Converted supra-adjacency matrix to CSR format.")

    # Compute numerator: tr(A^2 * A)
    print("Computing numerator...")
    num_start = time()
    numerator_step_1 = supra_adjacency_matrix @ supra_adjacency_matrix
    print("Numerator step 1 done... in", time() - num_start, " seconds")
    start = time()
    numerator_step_2 = numerator_step_1 @ supra_adjacency_matrix
    print("Numerator step 2 done... in ", time() - start, " seconds")
    start = time()
    numerator = numerator_step_2.diagonal().sum()
    print("Numerator step 3 done... in ", time() - start, " seconds")
    # Compute the F matrix: Ones() - Eye()
    print("Numerator done... in", time() - num_start, " seconds")

    print("Computing denominator (memory-efficient)...")
    den_start = time()

    N = n * l
    ones_vec = np.ones(N, dtype=np.float32)
    print(f"Ones vector created with shape: {ones_vec.shape}")

    # Ensure A is CSR for efficient matvec and elementwise ops
    A = supra_adjacency_matrix.tocsr()

    # u = A * 1, v = A^T * 1
    u = A.dot(ones_vec)
    v = A.T.dot(ones_vec)

    # tr(A J A) = v^T u
    tr_AJA = float(v.dot(u))

    # tr(A^2) = sum of elementwise product A * A.T
    tr_A2 = float((A.multiply(A.T)).sum())

    """
    F = J - I, with J = 1 1^T

    den = tr(AFA) = tr(AJA) - tr(AIA) = tr(AJA) - tr(A^2)
    tr(AJA) = 1^T (A^T A) 1 = (A^T 1)^T (A 1) = v^T u
    tr(AIA) = tr(A^2) = sum(A * A^T)

    den = v^T u - sum(A * A^T)
    """

    denominator = tr_AJA - tr_A2
    print("Denominator done... in", time() - den_start, "seconds")
    print(f"Denominator components: tr(AJA)={tr_AJA}, tr(A^2)={tr_A2}")

    return numerator / (max(supra_adjacency_matrix.data) * denominator) if denominator != 0 else 0.0


def compute_local_clustering_coefficient(supra_adjacency_matrix: torch.Tensor, node_index: int, layer_index: int, n: int, l: int) -> float:
    f_matrix = torch.ones(n*l, n*l, dtype=torch.float32, device=supra_adjacency_matrix.device) - torch.eye(n*l, dtype=torch.float32, device=supra_adjacency_matrix.device)
    #
    pass

if __name__ == "__main__":
    n, l = 10, 3
    matrix_size = n * l  # 30x30 matrix
    
    # Define a comprehensive set of non-zero entries for realistic testing
    # This includes intra-layer, inter-layer, and various weight patterns
    entries = [
        # Layer 0 intra-connections (nodes 0-9 in layer 0 -> indices 0,3,6,9,12,15,18,21,24,27)
        (0,1,1),(0,2,1),(0,4,1),(0,5,1),(0,7,1),
        (1,0,1),(1,3,1),(1,6,1),
        (2,0,1),(2,3,1),(2,4,1),(2,9,1),
        (3,1,1),(3,2,1),(3,4,1),(3,5,1),(3,7,1),(3,8,1),(3,9,1),
        (4,0,1),(4,2,1),(4,3,1),(4,5,1),(4,6,1),(4,7,1),
        (5,0,1),(5,3,1),(5,4,1),(5,6,1),(5,8,1),(5,9,1),
        (6,1,1),(6,4,1),(6,5,1),(6,8,1),(6,9,1),
        (7,0,1),(7,3,1),(7,4,1),
        (8,3,1),(8,5,1),(8,6,1),(8,9,1),
        (9,2,1),(9,3,1),(9,5,1),(9,6,1),(9,8,1),
        # Layer 1 intra-connections (nodes 0-9 in layer 1 -> indices 1,4,7,10,13,16,19,22,25,28)
        (10,11,1),(10,13,1),(10,17,1),
        (11,10,1),(11,13,1),(11,16,1),(11,19,1),
        (12,13,1),(12,14,1),(12,16,1),(12,18,1),(12,19,1),
        (13,10,1),(13,11,1),(13,12,1),(13,14,1),(13,16,1),(13,18,1),
        (14,12,1),(14,13,1),(14,15,1),(14,16,1),
        (15,14,1),(15,17,1),(15,19,1),
        (16,11,1),(16,12,1),(16,13,1),(16,14,1),(16,17,1),(16,19,1),
        (17,10,1),(17,15,1),(17,16,1),(17,18,1),
        (18,12,1),(18,13,1),(18,17,1),(18,19,1),
        (19,11,1),(19,12,1),(19,15,1),(19,16,1),(19,18,1),
        # Layer 2 intra-connections (nodes 0-9 in layer 2 -> indices 2,5,8,11,14,17,20,23,26,29)
        (20,21,1),(20,23,1),(20,25,1),(20,26,1),(20,28,1),
        (21,20,1),(21,22,1),(21,24,1),(21,25,1),(21,26,1),(21,27,1),
        (22,21,1),(22,23,1),(22,25,1),(22,27,1),(22,28,1),
        (23,20,1),(23,22,1),
        (24,21,1),(24,26,1),(24,28,1),
        (25,20,1),(25,21,1),(25,22,1),(25,28,1),(25,29,1),
        (26,20,1),(26,21,1),(26,24,1),
        (27,21,1),(27,22,1),
        (28,20,1),(28,22,1),(28,24,1),(28,25,1),
        (29,25,1),
        # Inter-layer connections (node i in layer j to node i in layer k)
        # Node 0: layer connections
        (0,10,1),(0,20,1),(10,0,1),(20,0,1),(10,20,1),(20,10,1),
        # Node 1: layer connections
        (1,11,1),(1,21,1),(11,1,1),(21,1,1),(11,21,1),(21,11,1),
        # Node 2: layer connections
        (2,12,1),(2,22,1),(12,2,1),(22,2,1),(12,22,1),(22,12,1),
        # Node 3: layer connections
        (3,13,1),(3,23,1),(13,3,1),(23,3,1),(13,23,1),(23,13,1),
        # Node 4: layer connections
        (4,14,1),(4,24,1),(14,4,1),(24,4,1),(14,24,1),(24,14,1),
        # Node 5: layer connections
        (5,15,1),(5,25,1),(15,5,1),(25,5,1),(15,25,1),(25,15,1),
        # Node 6: layer connections
        (6,16,1),(6,26,1),(16,6,1),(26,6,1),(16,26,1),(26,16,1),
        # Node 7: layer connections
        (7,17,1),(7,27,1),(17,7,1),(27,7,1),(17,27,1),(27,17,1),
        # Node 8: layer connections
        (8,18,1),(8,28,1),(18,8,1),(28,8,1),(18,28,1),(28,18,1),
        # Node 9: layer connections
        (9,19,1),(9,29,1),(19,9,1),(29,9,1),(19,29,1),(29,19,1),
    ]
    
    # Separate indices and values
    indices = torch.tensor([[entry[0] for entry in entries],
                        [entry[1] for entry in entries]], dtype=torch.long)
    values = torch.tensor([entry[2] for entry in entries], dtype=torch.float32)
    
    # Create sparse COO tensor
    sparse_matrix = torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=(matrix_size, matrix_size),
        dtype=torch.float32
    ).coalesce()

    # Test single metric
    nodes_katz_centrality = compute_katz_centrality(sparse_matrix, n, l)
    print("Katz centrality for each node:", nodes_katz_centrality)
