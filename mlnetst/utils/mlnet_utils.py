import torch
import networkx as nx

def build_tensor_from_supra_adjacency_matrix(supra_matrix, n, l):
    """
    Build a tensor from a supra-adjacency matrix.

    Args:
        supra_matrix (torch.Tensor): Input tensor of shape (N * L, N * L).
        n (int): Number of nodes.
        l (int): Number of layers.

    Returns:
        torch.Tensor: Tensor of shape (N, L, N, L).
    """
    return supra_matrix.view(n, l, n, l)

def build_supra_adjacency_matrix_from_tensor(t):
    """
    Build a supra-adjacency matrix from a tensor of mlnetsparse interactions.
    (N x L x N x L) -> (N * L x N * L)
    
    Args:
        t (torch.Tensor): Input tensor of shape (N, L, N, L).
    
    Returns:
        torch.Tensor: Supra-adjacency matrix of shape (N * L, N * L)
    """
    n, l = t.shape[0], t.shape[1]
    # Sanity check that the tensor is 4D
    if len(t.shape) != 4 or t.shape[0] != n or t.shape[2] != n or t.shape[1] != l or t.shape[3] != l:
        raise ValueError(f"Input tensor must be of shape ({n}, {l}, {n}, {l}), but got {t.shape}")
    # Reshape the tensor to a supra-adjacency matrix
    matrix = torch.zeros(n * l, n * l, device=t.device)
    for l_i in range(l):
        for l_j in range(l):
            matrix[n * l_i:n * (l_i + 1), n * l_j:n * (l_j + 1)] = t[:, l_i, :, l_j]
    print(f"TRUE: \n{matrix}")
    matrix = t.permute(1, 0, 3, 2).reshape(n * l, n * l)
    return matrix
    
def binarize_matrix(matrix):
    """
    Binarize a matrix by setting all non-zero elements to 1.

    Args:
        matrix (torch.Tensor): Input tensor of shape (N, N).

    Returns:
        torch.Tensor: Binarized tensor of shape (N, N).
    """
    if not isinstance(matrix, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    
    return (matrix != 0).to(torch.float32)  # Convert non-zero elements to 1.0

def get_aggregate_from_tensor(t):
    """
    Get the aggregate from a tensor of mlnetsparse interactions.

    Args:
        t (torch.Tensor): Input tensor of shape (N, L, N, L).

    Returns:
        torch.Tensor: Aggregated tensor of shape (N, N).
    """
    # aggregate is the sum over the second and fourth dimension,
    return t.sum(dim=(1, 3))

def get_aggregate_from_supra_adjacency_matrix(supra_matrix, n, l):
    """
    Get the aggregate from a supra-adjacency matrix.

    Args:
        supra_matrix (torch.Tensor): Input tensor of shape (N * L, N * L).
        n (int): Number of nodes.
        l (int): Number of layers.

    Returns:
        torch.Tensor: Aggregated tensor of shape (N, N).
    """
    t = build_tensor_from_supra_adjacency_matrix(supra_matrix, n, l)
    return get_aggregate_from_tensor(t)

def get_aggregate_network_from_supra_adjacency_matrix(supra_matrix, n, l):
    """
    Get the aggregate network from a supra-adjacency matrix.

    Args:
        supra_matrix (torch.Tensor): Input tensor of shape (N * L, N * L).
        n (int): Number of nodes.
        l (int): Number of layers.

    Returns:
        torch.Tensor: Aggregate network of shape (N, N).
    """
    aggregate = get_aggregate_from_supra_adjacency_matrix(supra_matrix, n, l)
    # Sanity check that the aggregate is 2D
    if len(aggregate.shape) != 2 or aggregate.shape[0] != n or aggregate.shape[1] != n:
        raise ValueError(f"Aggregate must be of shape ({n}, {n}), but got {aggregate.shape}")

    # Build the aggregate network as a graph
    G = nx.from_numpy_array(aggregate.detach().cpu().numpy(), create_using=nx.Graph)
    return G

def solve_eigenproblem(matrix):
    """
    Solve the eigenproblem for a given matrix.

    Args:
        matrix (torch.Tensor): Input matrix of shape (N, N).

    Returns:
        torch.Tensor: Eigenvalues of the matrix.
        torch.Tensor: Eigenvectors of the matrix.
    """
    if not isinstance(matrix, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input must be a square matrix of shape (N, N)")
    
    eigenvalues, eigenvectors = torch.linalg.eig(matrix)
    return eigenvalues, eigenvectors

def get_largest_eigenvalue(matrix):
    """
    Get the largest eigenvalue of a given matrix.

    Args:
        matrix (torch.Tensor): Input matrix of shape (N, N).

    Returns:
        torch.Tensor: Largest eigenvalue of the matrix.
    """
    eigenvalues, _ = solve_eigenproblem(matrix)
    return torch.max(eigenvalues.real)  # Return the largest real part of the eigenvalues

def get_canonical_vector(n, i):
    """
    Return the ith canonical vector of size n.

    Args:
        n (int): Size of the vector.
        i (int): Index of the canonical vector.

    Returns:
        torch.Tensor: The ith canonical vector of size n.
    """
    if i < 0 or i >= n:
        raise ValueError(f"Index i must be between 0 and {n-1}, but got {i}")
    vec = torch.zeros(1,n)
    vec[0, i] = 1.0
    return vec

def get_laplacian_matrix(matrix):
    """
    Compute the Laplacian matrix of a given matrix.

    Args:
        matrix (torch.Tensor): Input matrix of shape (N, N).

    Returns:
        torch.Tensor: Laplacian matrix of the input matrix.
    """
    if not isinstance(matrix, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input must be a square matrix of shape (N, N)")
    
    degree_matrix = torch.diag(torch.sum(matrix, dim=1))
    laplacian_matrix = degree_matrix - matrix
    return laplacian_matrix

def get_normalized_laplacian_matrix(matrix):
    """
    Compute the normalized Laplacian matrix of a given matrix.
    
    The normalized lapacian matrix is a key concept in spectral graph theory. For an adjacency matrix A, the norm laplacian norm is tipycally:
    L = I - D^(-1/2) * A * D^(-1/2)

    Args:
        matrix (torch.Tensor): Input matrix of shape (N, N).

    Returns:
        torch.Tensor: Normalized Laplacian matrix of the input matrix.
    """
    if not isinstance(matrix, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input must be a square matrix")
    
    # Compute degree matrix
    degrees = torch.sum(matrix, dim=1)
    
    # Handle disconnected nodes (degree = 0)
    # Set their degree to 1 to avoid division by zero
    degrees = torch.where(degrees > 0, degrees, torch.ones_like(degrees))
    
    # Compute D^(-1/2)
    d_inv_sqrt = torch.diag(1.0 / torch.sqrt(degrees))
    
    # Compute normalized Laplacian: I - D^(-1/2)AD^(-1/2)
    normalized_laplacian = torch.eye(matrix.shape[0], device=matrix.device) - \
                        d_inv_sqrt @ matrix @ d_inv_sqrt
    
    # Set rows/columns corresponding to disconnected nodes to zero
    disconnected_mask = degrees == 1  # identifies originally zero degrees
    normalized_laplacian[disconnected_mask] = 0
    normalized_laplacian[:, disconnected_mask] = 0
    
    # Set diagonal entries for disconnected nodes to 1
    normalized_laplacian[disconnected_mask, disconnected_mask] = 1
    
    return normalized_laplacian

def build_density_matrix_bgs(matrix):
    """
    Build a density matrix from a given matrix using the BGS method.

    Args:
        matrix (torch.Tensor): Input matrix of shape (N, N).

    Returns:
        torch.Tensor: Density matrix of the input matrix.
    """
    if not isinstance(matrix, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input must be a square matrix of shape (N, N)")

    density_matrix = get_laplacian_matrix(matrix)
    # Normalize the density matrix
    return density_matrix / torch.diag(density_matrix)


if __name__ == "__main__":
    # Example usage
    t = torch.randn(3, 1, 3, 1)  # Example tensor of shape (3, 2, 3, 2)
    print(t)
    supra_matrix = build_supra_adjacency_matrix_from_tensor(t)
    print(supra_matrix.shape)  # Should print (6, 6) for this example
    print(supra_matrix)