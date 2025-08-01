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

def compute_indegree(supra_adjacency_matrix, n, l):
    """
    Compute the indegree of each node, considering them as replica nodes in a multilayer network.

    It is required to know how many nodes and layers are in the network to correctly compute the indegree from the supraadjacency matrix.
    Args:
        adjacency_matrix (torch.Tensor): The supra-adjacency matrix of shape (N * L, N * L).
        n (int): Number of nodes.
        l (int): Number of layers.
    Returns:
    
    """
    
    # mask the supra-adjacency matrix to keep only the diagonal blocks
    diagonal_blocks_mask = get_diagonal_blocks(n, l, device=supra_adjacency_matrix.device)
    
    bin_supra_adjacency_matrix = binarize_matrix(supra_adjacency_matrix)
    # apply the mask
    masked_matrix = bin_supra_adjacency_matrix * diagonal_blocks_mask
    
    # Compute the indegree (sum along columns)
    layers_indegrees = masked_matrix.sum(dim=0)
    
    if layers_indegrees.is_sparse:
        # Handle sparse tensor
        indices = layers_indegrees.indices()[0]  # Get indices
        values = layers_indegrees.values()       # Get values
        
        # Create a dense tensor to accumulate results
        nodes_indegrees = torch.zeros(n, device=supra_adjacency_matrix.device)
        
        # Sum values for corresponding nodes (using modulo)
        node_indices = indices % n
        nodes_indegrees.index_add_(0, node_indices, values)
    else:
        # Handle dense tensor - reshape and sum across layers
        nodes_indegrees = layers_indegrees.reshape(l, n).sum(dim=0)
    
    return nodes_indegrees

def compute_instrength(supra_adjacency_matrix, n, l):
    """
    Compute the indegree of each node, considering them as replica nodes in a multilayer network.

    It is required to know how many nodes and layers are in the network to correctly compute the instrength from the supraadjacency matrix.
    Args:
        adjacency_matrix (torch.Tensor): The supra-adjacency matrix of shape (N * L, N * L).
        n (int): Number of nodes.
        l (int): Number of layers.
    Returns:
    
    """
    
    # mask the supra-adjacency matrix to keep only the diagonal blocks
    diagonal_blocks_mask = get_diagonal_blocks(n, l, device=supra_adjacency_matrix.device)
    
    
    # apply the mask
    masked_matrix = supra_adjacency_matrix * diagonal_blocks_mask

    # Compute the instrength (sum along columns)
    layers_instrengths = masked_matrix.sum(dim=0)

    if layers_instrengths.is_sparse:
        # Handle sparse tensor
        indices = layers_instrengths.indices()[0]  # Get indices
        values = layers_instrengths.values()       # Get values

        # Create a dense tensor to accumulate results
        nodes_instrengths = torch.zeros(n, device=supra_adjacency_matrix.device)
        
        # Sum values for corresponding nodes (using modulo)
        node_indices = indices % n
        nodes_instrengths.index_add_(0, node_indices, values)
    else:
        # Handle dense tensor - reshape and sum across layers
        nodes_instrengths = layers_instrengths.reshape(l, n).sum(dim=0)

    return nodes_instrengths

def compute_outdegree(supra_adjacency_matrix, n, l):
    """
    Compute the outdegree of each node, considering them as replica nodes in a multilayer network.

    Args:
        supra_adjacency_matrix (torch.Tensor): The supra-adjacency matrix of shape (N * L, N * L).
        n (int): Number of nodes.
        l (int): Number of layers.

    Returns:
        torch.Tensor: A tensor of shape (N * L) containing the outdegree of each node.
    """
    bin_supra_adjacency_matrix = binarize_matrix(supra_adjacency_matrix)

    # mask the supra-adjacency matrix to keep only the diagonal blocks
    diagonal_blocks_mask = get_diagonal_blocks(n, l, device=supra_adjacency_matrix.device)

    # apply the mask
    masked_matrix = bin_supra_adjacency_matrix * diagonal_blocks_mask

    # compute the outdegree (sum along rows)
    layers_outdegrees = masked_matrix.sum(dim=1)
    if layers_outdegrees.is_sparse:
        # Handle sparse tensor
        indices = layers_outdegrees.indices()[0]
        values = layers_outdegrees.values()
        # Create a dense tensor to accumulate results
        nodes_outdegrees = torch.zeros(n, device=supra_adjacency_matrix.device)
        # Sum values for corresponding nodes (using modulo)
        node_indices = indices % n
        nodes_outdegrees.index_add_(0, node_indices, values)
    else:
        # Handle dense tensor - reshape and sum across layers
        nodes_outdegrees = layers_outdegrees.reshape(l, n).sum(dim=0)
    
    # consider that every n elements of the outdegrees correspond to a single node
    return nodes_outdegrees
    

def compute_multi_indegree(supra_adjacency_matrix, n, l):
    """
    Compute the multiindegree of each node, considering them as replica nodes in a multilayer network. 
    This function computes the indegree for each node across all layers.

    Args:
        supra_adjacency_matrix (torch.Tensor): The supra-adjacency matrix of shape (N * L, N * L).
        n (int): Number of nodes.
        l (int): Number of layers.

    Returns:
        torch.Tensor: A tensor of shape (N * L) containing the multiindegree of each node.
    """
    bin_supra_adjacency_matrix = binarize_matrix(supra_adjacency_matrix)

    # mask the supra-adjacency matrix to keep only the diagonal blocks
    non_diagonal_blocks_mask = get_non_diagonal_blocks(n, l, device=supra_adjacency_matrix.device)

    # apply the mask
    masked_matrix = bin_supra_adjacency_matrix * non_diagonal_blocks_mask

    # compute the multiindegree (sum along columns)
    layers_multiindegrees = masked_matrix.sum(dim=0)
    if layers_multiindegrees.is_sparse:
        # Handle sparse tensor
        indices = layers_multiindegrees.indices()[0]
        values = layers_multiindegrees.values()
        # Create a dense tensor to accumulate results
        nodes_multiindegrees = torch.zeros(n, device=supra_adjacency_matrix.device)
        # Sum values for corresponding nodes (using modulo)
        node_indices = indices % n
        nodes_multiindegrees.index_add_(0, node_indices, values)
    else:
        # Handle dense tensor - reshape and sum across layers
        nodes_multiindegrees = layers_multiindegrees.reshape(l, n).sum(dim=0)
    
    # consider that every n elements of the multiindegrees correspond to a single node
    return nodes_multiindegrees
    
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
    