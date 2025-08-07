import pytest
import torch
import numpy as np
from mlnetst.utils.mlnet_utils import (
    build_supra_adjacency_matrix_from_tensor,
    build_tensor_from_supra_adjacency_matrix,
)

from mlnetst.utils.mlnet_metrics_utils import (
    compute_indegree,
    compute_instrength,
    compute_outdegree,
    compute_outstrength,
    compute_multi_indegree,
    compute_multi_instrength,
    compute_multi_outdegree,
    compute_multi_outstrength,
)


@pytest.fixture
def sample_adjacency_matrix():
    """
    Create a custom sparse COO adjacency matrix with predefined structure.
    This version creates the sparse matrix from scratch without going through dense format.
    
    Matrix structure (6x6 for n=3, l=2):
    Node structure: [0_0, 0_1, 1_0, 1_1, 2_0, 2_1] where i_j means node i in layer j
    """
    n, l = 3, 2
    matrix_size = n * l  # 6x6 matrix
    
    # Define non-zero entries directly (row, col, value)
    # This simulates a realistic sparse adjacency matrix
    entries = [
        # Node 0 connections (rows 0-1)
        (0,2,1.0), (0,5,0.5),
        (1,0,2.0),
        (2,4,1.0),
        (3,4,2.0),
        (4,2,0.5),
        (5,0,3.0),
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
    
    return {
        'matrix': sparse_matrix,
        'n': n,
        'l': l,
        'nnz': sparse_matrix._nnz(),
        'density': sparse_matrix._nnz() / (matrix_size * matrix_size),
        'entries': entries  # Original entries for reference
    }

@pytest.fixture
def expected_degrees():
    """
    Fixture providing expected in/out degrees for sample_adjacency_matrix.
    
    Calculations based on the adjacency matrix structure:
    - Regular degree: sum across all layers for each node
    - Multi-degree: count of inter-layer connections for each node
    """
    return {
        # In-degrees: incoming edges to each node (sum across both layers)
        'in_degrees': torch.tensor([1, 1, 1], dtype=torch.float),
        'in_strengths': torch.tensor([2.0, 2.0, 1.0], dtype=torch.float),
        
        # Out-degrees: outgoing edges from each node (sum across both layers)  
        'out_degrees': torch.tensor([2, 1, 0], dtype=torch.float),
        'out_strengths': torch.tensor([3.0, 2.0, 0.0], dtype=torch.float),
        
        # Multi-indegrees: inter-layer incoming connections
        # This counts connections FROM other layers TO each node
        'multi_in_degrees': torch.tensor([1, 1, 2], dtype=torch.float),
        'multi_in_strengths': torch.tensor([3.0, 1.0, 1.0], dtype=torch.float),
        # Multi-outdegrees: inter-layer outgoing connections  
        # This counts connections TO other layers FROM each node
        'multi_out_degrees': torch.tensor([1, 1, 2], dtype=torch.float),
        'multi_out_strengths': torch.tensor([0.5, 0.5, 4.0], dtype=torch.float)
    }


@pytest.fixture
def dense_comparison_matrix(sample_adjacency_matrix):
    """
    Create dense version of the sparse matrix for easier manual verification.
    """
    sparse_matrix = sample_adjacency_matrix['matrix']
    dense_matrix = sparse_matrix.to_dense()
    return {
        'dense': dense_matrix,
        'n': sample_adjacency_matrix['n'],
        'l': sample_adjacency_matrix['l']
    }


def test_sparse_matrix_properties(sample_adjacency_matrix):
    """
    Test that the sparse matrix has expected properties.
    """
    matrix_data = sample_adjacency_matrix
    sparse_matrix = matrix_data['matrix']
    n, l = matrix_data['n'], matrix_data['l']
    
    # Basic sparse matrix properties
    assert sparse_matrix.is_sparse, "Matrix should be sparse"
    assert sparse_matrix.layout == torch.sparse_coo, "Matrix should be in COO format"
    assert sparse_matrix.size() == (n*l, n*l), f"Expected size ({n*l}, {n*l}), got {sparse_matrix.size()}"
    assert sparse_matrix.is_coalesced(), "Matrix should be coalesced"
    
    # Check that we have the expected number of non-zero entries
    expected_nnz = len(matrix_data['entries'])
    assert sparse_matrix._nnz() == expected_nnz, f"Expected {expected_nnz} non-zeros, got {sparse_matrix._nnz()}"
    
    # Verify sparsity
    total_elements = (n * l) ** 2
    sparsity_ratio = sparse_matrix._nnz() / total_elements
    assert 0 < sparsity_ratio < 1, f"Matrix should be sparse but not empty, sparsity: {sparsity_ratio}"


def test_matrix_conversion_consistency(sample_adjacency_matrix):
    """
    Test that sparse-to-dense-to-sparse conversion preserves the matrix.
    """
    original_sparse = sample_adjacency_matrix['matrix']
    
    # Convert to dense and back to sparse
    dense_version = original_sparse.to_dense()
    sparse_again = dense_version.to_sparse_coo()
    
    # Compare values (need to sort since COO format may have different ordering)
    original_values, original_indices = torch.sort(original_sparse.values())
    new_values, new_indices = torch.sort(sparse_again.values())
    
    assert torch.allclose(original_values, new_values, atol=1e-6), "Values should be preserved in conversion"
    assert original_sparse._nnz() == sparse_again._nnz(), "Number of non-zeros should be preserved"


def test_compute_indegree(sample_adjacency_matrix, expected_degrees):
    """
    Test the computation of indegree from a sparse adjacency matrix.
    """
    indegree = compute_indegree(sample_adjacency_matrix["matrix"], 
                                n=sample_adjacency_matrix['n'], 
                                l=sample_adjacency_matrix['l'])
    
    expected = expected_degrees['in_degrees']
    assert indegree.shape == expected.shape, f"Shape mismatch: got {indegree.shape}, expected {expected.shape}"
    assert torch.allclose(indegree, expected, atol=1e-5), f"In-degree mismatch: got {indegree}, expected {expected}"


def test_compute_outdegree(sample_adjacency_matrix, expected_degrees):
    """
    Test the computation of outdegree from a sparse adjacency matrix.
    """
    outdegree = compute_outdegree(sample_adjacency_matrix["matrix"], 
                                n=sample_adjacency_matrix['n'], 
                                l=sample_adjacency_matrix['l'])
    
    expected = expected_degrees['out_degrees']
    assert outdegree.shape == expected.shape, f"Shape mismatch: got {outdegree.shape}, expected {expected.shape}"
    assert torch.allclose(outdegree, expected, atol=1e-5), f"Out-degree mismatch: got {outdegree}, expected {expected}"


def test_compute_multi_indegree(sample_adjacency_matrix, expected_degrees):
    """
    Test the computation of multi-indegree from a sparse adjacency matrix.
    Multi-indegree counts inter-layer incoming connections.
    """
    multi_indegree = compute_multi_indegree(sample_adjacency_matrix["matrix"], 
                                            n=sample_adjacency_matrix['n'], 
                                            l=sample_adjacency_matrix['l'])
    
    expected = expected_degrees['multi_in_degrees']
    assert multi_indegree.shape == expected.shape, f"Shape mismatch: got {multi_indegree.shape}, expected {expected.shape}"
    assert torch.allclose(multi_indegree, expected, atol=1e-5), f"Multi-in-degree mismatch: got {multi_indegree}, expected {expected}"


def test_compute_multi_outdegree(sample_adjacency_matrix, expected_degrees):
    """
    Test the computation of multi-outdegree from a sparse adjacency matrix.
    Multi-outdegree counts inter-layer outgoing connections.
    """
    multi_outdegree = compute_multi_outdegree(sample_adjacency_matrix["matrix"], 
                                            n=sample_adjacency_matrix['n'], 
                                            l=sample_adjacency_matrix['l'])
    
    expected = expected_degrees['multi_out_degrees']
    assert multi_outdegree.shape == expected.shape, f"Shape mismatch: got {multi_outdegree.shape}, expected {expected.shape}"
    assert torch.allclose(multi_outdegree, expected, atol=1e-5), f"Multi-out-degree mismatch: got {multi_outdegree}, expected {expected}"

def test_compute_instrengths(sample_adjacency_matrix, expected_degrees):
    """
    Test the computation of in-strengths from a sparse adjacency matrix.
    """
    instrengths = compute_instrength(sample_adjacency_matrix["matrix"], 
                                    n=sample_adjacency_matrix['n'], 
                                    l=sample_adjacency_matrix['l'])
    
    expected = expected_degrees['in_strengths']
    assert instrengths.shape == expected.shape, f"Shape mismatch: got {instrengths.shape}, expected {expected.shape}"
    assert torch.allclose(instrengths, expected, atol=1e-5), f"In-strength mismatch: got {instrengths}, expected {expected}"


def test_degree_computation_with_dense_matrix(dense_comparison_matrix, expected_degrees):
    """
    Test that degree computations work the same with dense matrices.
    This serves as a cross-validation of our sparse implementation.
    """
    dense_matrix = dense_comparison_matrix['dense']
    n, l = dense_comparison_matrix['n'], dense_comparison_matrix['l']
    
    # Convert dense back to sparse for testing
    sparse_from_dense = dense_matrix.to_sparse_coo()
    
    # Compute degrees using the converted matrix
    indegree = compute_indegree(sparse_from_dense, n=n, l=l)
    outdegree = compute_outdegree(sparse_from_dense, n=n, l=l)
    multi_indegree = compute_multi_indegree(sparse_from_dense, n=n, l=l)
    multi_outdegree = compute_multi_outdegree(sparse_from_dense, n=n, l=l)
    
    # Should match expected values
    assert torch.allclose(indegree, expected_degrees['in_degrees'], atol=1e-5)
    assert torch.allclose(outdegree, expected_degrees['out_degrees'], atol=1e-5) 
    assert torch.allclose(multi_indegree, expected_degrees['multi_in_degrees'], atol=1e-5)
    assert torch.allclose(multi_outdegree, expected_degrees['multi_out_degrees'], atol=1e-5)


def test_edge_cases():
    """
    Test edge cases like empty matrices, single node, etc.
    """
    # Test with minimal matrix (1 node, 1 layer)
    n, l = 1, 1
    indices = torch.tensor([[], []], dtype=torch.long)
    values = torch.tensor([], dtype=torch.float32)
    
    empty_sparse = torch.sparse_coo_tensor(
        indices=indices,
        values=values, 
        size=(n*l, n*l),
        dtype=torch.float32
    ).coalesce()
    
    # All degrees should be zero
    indegree = compute_indegree(empty_sparse, n=n, l=l)
    outdegree = compute_outdegree(empty_sparse, n=n, l=l)
    multi_indegree = compute_multi_indegree(empty_sparse, n=n, l=l) 
    multi_outdegree = compute_multi_outdegree(empty_sparse, n=n, l=l)
    
    assert torch.allclose(indegree, torch.zeros(n)), "Empty matrix should have zero in-degree"
    assert torch.allclose(outdegree, torch.zeros(n)), "Empty matrix should have zero out-degree"
    assert torch.allclose(multi_indegree, torch.zeros(n)), "Empty matrix should have zero multi-in-degree"
    assert torch.allclose(multi_outdegree, torch.zeros(n)), "Empty matrix should have zero multi-out-degree"

@pytest.mark.parametrize("n,l", [(2, 3), (4, 2), (3, 3), (5, 1)])
def test_different_network_sizes(n, l):
    """
    Test degree computation with different network sizes.
    """
    matrix_size = n * l
    
    # Create a simple diagonal matrix with some off-diagonal elements
    row_indices = list(range(matrix_size)) + [0, 1, 2][:min(3, matrix_size-1)]
    col_indices = list(range(matrix_size)) + [1, 2, 0][:min(3, matrix_size-1)]
    values = [1.0] * len(row_indices)
    
    indices = torch.tensor([row_indices, col_indices], dtype=torch.long)
    values_tensor = torch.tensor(values, dtype=torch.float32)
    
    sparse_matrix = torch.sparse_coo_tensor(
        indices=indices,
        values=values_tensor,
        size=(matrix_size, matrix_size),
        dtype=torch.float32
    ).coalesce()
    
    # Should not raise any errors
    indegree = compute_indegree(sparse_matrix, n=n, l=l)
    outdegree = compute_outdegree(sparse_matrix, n=n, l=l)
    multi_indegree = compute_multi_indegree(sparse_matrix, n=n, l=l)
    multi_outdegree = compute_multi_outdegree(sparse_matrix, n=n, l=l)
    
    # Basic shape checks
    assert indegree.shape == (n,), f"In-degree shape should be ({n},), got {indegree.shape}"
    assert outdegree.shape == (n,), f"Out-degree shape should be ({n},), got {outdegree.shape}"
    assert multi_indegree.shape == (n,), f"Multi-in-degree shape should be ({n},), got {multi_indegree.shape}"
    assert multi_outdegree.shape == (n,), f"Multi-out-degree shape should be ({n},), got {multi_outdegree.shape}"
    
    # All values should be non-negative
    assert torch.all(indegree >= 0), "In-degrees should be non-negative"
    assert torch.all(outdegree >= 0), "Out-degrees should be non-negative" 
    assert torch.all(multi_indegree >= 0), "Multi-in-degrees should be non-negative"
    assert torch.all(multi_outdegree >= 0), "Multi-out-degrees should be non-negative"