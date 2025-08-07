import torch
import pymnet
from mlnetst.utils.mlnet_utils import get_aggregate_from_supra_adjacency_matrix, build_supra_adjacency_matrix_from_tensor, binarize_matrix

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

def compute_indegree(supra_adjacency_matrix: torch.Tensor, n: int, l: int) -> torch.Tensor:
    """
    Compute indegree, which is the number of incoming edges aggregated across all intralayers, therefore it requires only diagonal blocks.
    
    Args:
        supra_adjacency_matrix: Sparse tensor of shape (N*L, N*L)
        n: Number of nodes per layer
        l: Number of layers
        
    Returns:
        Tensor of shape (N,) containing indegrees
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

def compute_total_degree(supra_adjacency_matrix, n, l):
    """
    Compute the total degree of each node, considering them as replica nodes in a multilayer network.
    The total degree is the sum of indegree and outdegree.

    Args:
        supra_adjacency_matrix (torch.Tensor): The supra-adjacency matrix of shape (N * L, N * L).
        n (int): Number of nodes.
        l (int): Number of layers.

    Returns:
        torch.Tensor: A tensor of shape (N * L) containing the total degree of each node.
    """
    indegree = compute_indegree(supra_adjacency_matrix, n, l)
    outdegree = compute_outdegree(supra_adjacency_matrix, n, l)
    multi_indegree = compute_multi_indegree(supra_adjacency_matrix, n, l)
    multi_outdegree = compute_multi_outdegree(supra_adjacency_matrix, n, l)
    return indegree + outdegree + multi_indegree + multi_outdegree

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
    
def compute_average_global_clustering(supra_adjacency_matrix: torch.Tensor, n: int, l: int) -> float:
    """
    Compute the average global clustering coefficient for a multilayer network as described in De Domenico et al. (2013).
    """

    f_matrix = torch.ones(n*l, n*l, dtype=torch.float32, device=supra_adjacency_matrix.device) - torch.eye(n*l, dtype=torch.float32, device=supra_adjacency_matrix.device)
    # compute the trace of A*A*A
    num = torch.trace(supra_adjacency_matrix @ supra_adjacency_matrix @ supra_adjacency_matrix)
    # compute the trace of A*F*A
    den = torch.trace(supra_adjacency_matrix @ f_matrix @ supra_adjacency_matrix)

    return num / (torch.max(supra_adjacency_matrix) * den)

def compute_local_clustering_coefficient(supra_adjacency_matrix: torch.Tensor, node_index: int, layer_index: int, n: int, l: int) -> float:
    f_matrix = torch.ones(n*l, n*l, dtype=torch.float32, device=supra_adjacency_matrix.device) - torch.eye(n*l, dtype=torch.float32, device=supra_adjacency_matrix.device)
    # 