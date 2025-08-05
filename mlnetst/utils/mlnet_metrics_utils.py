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

def is_in_diagonal_block(indices: torch.Tensor, n: int) -> torch.Tensor:
    """
    Check which indices belong to diagonal blocks.
    
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
    
    # Find which indices are in diagonal blocks
    diagonal_mask = is_in_diagonal_block(indices, n)
    
    # Filter indices and values
    diagonal_indices = indices[:, diagonal_mask]
    diagonal_values = values[diagonal_mask]
    
    # Create binary values for degree calculation
    binary_values = torch.ones_like(diagonal_values)
    
    # Compute node indices within layers
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
    nodes_instrengths = torch.zeros(n * l, device=supra_adjacency_matrix.device)
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
    nodes_outstrengths = torch.zeros(n * l, device=supra_adjacency_matrix.device)
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
    
def compute_multi_outdegree(supra_adjacency_matrix, n, l):
    """
    Compute the multioutdegree of each node, considering them as replica nodes in a multilayer network. 
    This function computes the outdegree for each node across all layers.

    Args:
        supra_adjacency_matrix (torch.Tensor): The supra-adjacency matrix of shape (N * L, N * L).
        n (int): Number of nodes.
        l (int): Number of layers.

    Returns:
        torch.Tensor: A tensor of shape (N * L) containing the multioutdegree of each node.
    """
    # mask the supra-adjacency matrix to keep only the diagonal blocks
    non_diagonal_blocks_mask = get_non_diagonal_blocks(n, l, device=supra_adjacency_matrix.device)

    # apply the mask
    masked_matrix = supra_adjacency_matrix * non_diagonal_blocks_mask

    # compute the multioutdegree (sum along rows)
    layers_multioutdegrees = masked_matrix.sum(dim=1)
    # consider that every n elements of the multioutdegrees correspond to a single node
    nodes_multioutdegrees = layers_multioutdegrees.reshape(l, n).sum(dim=0)
    return nodes_multioutdegrees

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

def compute_clustering_coefficient(net: pymnet.MultilayerNetwork):
    
    return pymnet.gcc_zhang(net)
    