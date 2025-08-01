import pytest
import torch
import numpy as np

from mlnetst.utils.mlnet_utils import (
    build_supra_adjacency_matrix_from_tensor,
    build_tensor_from_supra_adjacency_matrix,
)



@pytest.fixture
def sample_4d_tensor():
    """
    Create a sample 4D tensor for testing.
    Returns:
        torch.Tensor: A 4D tensor of shape (N, L, N, L).
    """
    n, l = 3, 2
    return torch.arange(n * l * n * l, dtype=torch.float32).reshape(n, l, n, l)

def test_supra_adjacency_matrix_conversion(sample_4d_tensor):
    """
    Test the conversion of a 4D tensor to a supra-adjacency matrix.
    """
    supra_matrix = build_supra_adjacency_matrix_from_tensor(sample_4d_tensor)
    # check shape
    assert supra_matrix.shape == (6, 6)
    
    # convert back to tensor
    reconstructed_tensor = build_tensor_from_supra_adjacency_matrix(supra_matrix, n=3, l=2)    
    
    # Check if reconstruction matches original
    assert torch.allclose(sample_4d_tensor, reconstructed_tensor, atol=1e-5)