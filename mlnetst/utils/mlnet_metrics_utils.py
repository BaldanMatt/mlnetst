import torch
from mlnetst.utils.mlnet_utils import get_aggregate_from_supra_adjacency_matrix, build_supra_adjacency_matrix_from_tensor, binarize_matrix

def compute_indegree(adjacency_matrix):
    """
    Compute the indegree of each node in a directed graph represented by an adjacency matrix.

    Args:
        adjacency_matrix (torch.Tensor): Adjacency matrix of shape (N, N).

    Returns:
        torch.Tensor: Indegree of each node.
    """
    if not isinstance(adjacency_matrix, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    
    if len(adjacency_matrix.shape) != 2 or adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        raise ValueError("Input must be a square matrix of shape (N, N)")
    
    return torch.sum(adjacency_matrix, dim=0)  # Sum along rows to get indegree

def compute_outdegree(adjacency_matrix):
    """
    Compute the outdegree of each node in a directed graph represented by an adjacency matrix.

    Args:
        adjacency_matrix (torch.Tensor): Adjacency matrix of shape (N, N).

    Returns:
        torch.Tensor: Outdegree of each node.
    """
    if not isinstance(adjacency_matrix, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    
    if len(adjacency_matrix.shape) != 2 or adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        raise ValueError("Input must be a square matrix of shape (N, N)")
    
    return torch.sum(adjacency_matrix, dim=1)  # Sum along columns to get outdegree

def get_multi_outdegree(matrix, n, l):
    """
    Return the muti-out-degree, not accounting for interlinks

    Args:
        matrix (torch.Tensor): the supra-adjacency matrix
        n (int): number of nodes in the supra-adjacency matrix
        l (int): number of layers in the supra-adjacency matrix

    Returns:
        torch.Tensor: the multi-out-degree of each node
    """
    aggregate = get_aggregate_from_supra_adjacency_matrix(binarize_matrix(matrix), n, l)
    multioutdegree = torch.sum(aggregate, dim=1)  # Sum along columns to get multi-outdegree

    return multioutdegree

def get_multi_indegree(matrix, n, l):
    """
    Return the multi-in-degree, not accounting for interlinks

    Args:
        matrix (torch.Tensor): the supra-adjacency matrix
        n (int): number of nodes in the supra-adjacency matrix
        l (int): number of layers in the supra-adjacency matrix

    Returns:
        torch.Tensor: the multi-in-degree of each node
    """
    aggregate = get_aggregate_from_supra_adjacency_matrix(binarize_matrix(matrix), n, l)
    multiindegree = torch.sum(aggregate, dim=0)  # Sum along rows to get multi-indegree

    return multiindegree


if __name__ == "__main__":
    # Example usage with 3 nodes and 2 layers
    n = 3
    l = 2
    # init 4d tensor
    torch.random.manual_seed(0)  # For reproducibility
    t = torch.randn(n, l, n, l)  # Example tensor of shape (3, 2, 3, 2)
    print("Input tensor shape:", t.shape)
    # Remove some interactions to manipulate degree
    t[0, 0, 1, 1] = 0  # Remove interaction between node 0 and node 1 in layer 0
    t[1, 0, 2, 1] = 0  # Remove interaction between node 1 and node 2 in layer 1
    t[2, 1, 1, 0] = 0
    
    supra_matrix = build_supra_adjacency_matrix_from_tensor(t)
    print("Supra-adjacency matrix shape:", supra_matrix.shape)  # Should print
    print(supra_matrix)
    
    
    print("Indegree:", compute_indegree(supra_matrix))
    print("Outdegree:", compute_outdegree(supra_matrix))
    print("Multi-outdegree:", get_multi_outdegree(supra_matrix, n, l))
    print("Multi-indegree:", get_multi_indegree(supra_matrix, n, l))
    