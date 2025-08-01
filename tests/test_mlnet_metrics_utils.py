import pytest
import torch
import numpy as np
from mlnetst.utils.mlnet_utils import (
    build_supra_adjacency_matrix_from_tensor,
    build_tensor_from_supra_adjacency_matrix,
)

from mlnetst.utils.mlnet_metrics_utils import (
    compute_indegree,
    compute_outdegree,
    compute_multi_indegree,
    compute_multi_outdegree,
)

@pytest.fixture
def sample_adjacency_matrix():
    """
    Create a sample adjacency matrix for testing.
    Returns:
        torch.Tensor: A square adjacency matrix of shape (N, N).
    """
    n, l = 3, 2
    # create a 4D tensor and convert it to a supra-adjacency matrix
    tensor = torch.arange(n * l * n * l, dtype=torch.float32).reshape(n, l, n, l)
    matrix = build_supra_adjacency_matrix_from_tensor(tensor)
    # fill in 5 zeros not in sequential positions to simulate a real adjacency matrix of NL x NL dimensions
    # Set specific positions to zero to create realistic sparsity pattern
    zero_positions = [
        (0,0), (0,3),(1,4),(2,5),(3,0),(4,1),(4,4),(5,4),(3,2)
    ]
    for i, j in zero_positions:
        matrix[i, j] = 0.0
    
    return {
        'matrix': matrix,
        'n': n,
        'l': l
    }

@pytest.fixture
def expected_degrees():
    """
    Fixture providing expected in/out degrees for sample_adjacency_matrix.
    """
    return {
        'in_degrees': torch.tensor([5, 4, 6], dtype=torch.long),
        'out_degrees': torch.tensor([5, 5, 5], dtype=torch.long),
        'multi_in_degrees': torch.tensor([4, 4, 4], dtype=torch.long),
        'multi_out_degrees': torch.tensor([3, 4, 5], dtype=torch.long)
    }
    
def test_compute_indegree(sample_adjacency_matrix, expected_degrees):
    """
    Test the computation of indegree from an adjacency matrix.
    """
    indegree = compute_indegree(sample_adjacency_matrix["matrix"], 
                                n=sample_adjacency_matrix['n'], 
                                l=sample_adjacency_matrix['l'])
    assert torch.allclose(indegree, expected_degrees['in_degrees'], atol=1e-5)
    
def test_compute_outdegree(sample_adjacency_matrix, expected_degrees):
    """
    Test the computation of outdegree from an adjacency matrix.
    """
    outdegree = compute_outdegree(sample_adjacency_matrix["matrix"], 
                                   n=sample_adjacency_matrix['n'], 
                                   l=sample_adjacency_matrix['l'])
    assert torch.allclose(outdegree, expected_degrees['out_degrees'], atol=1e-5)
    
def test_compute_multindegree(sample_adjacency_matrix, expected_degrees):
    """
    Test the computation of multi-indegree from an adjacency matrix.
    """
    multi_indegree = compute_multi_indegree(sample_adjacency_matrix["matrix"], 
                                            n=sample_adjacency_matrix['n'], 
                                            l=sample_adjacency_matrix['l'])
    # The expected multi-indegree is the same as the indegree in this case
    assert torch.allclose(multi_indegree, expected_degrees['multi_in_degrees'], atol=1e-5)
    
def test_compute_multioutdegree(sample_adjacency_matrix, expected_degrees):
    """
    Test the computation of multi-outdegree from an adjacency matrix.
    """
    multi_outdegree = compute_multi_outdegree(sample_adjacency_matrix["matrix"], 
                                               n=sample_adjacency_matrix['n'], 
                                               l=sample_adjacency_matrix['l'])
    # The expected multi-outdegree is the same as the outdegree in this case
    assert torch.allclose(multi_outdegree, expected_degrees['multi_out_degrees'], atol=1e-5)