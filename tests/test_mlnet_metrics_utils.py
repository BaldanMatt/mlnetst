import pytest
import torch
import numpy as np
from mlnetst.utils.mlnet_utils import (
    build_supra_adjacency_matrix_from_tensor,
    build_tensor_from_supra_adjacency_matrix,
)

import networkx as nx

from mlnetst.utils.mlnet_metrics_utils import (
    compute_indegree,
    compute_indegree_for_layer,
    compute_average_global_clustering,
    compute_instrength,
    compute_outdegree,
    compute_outstrength,
    compute_multi_indegree,
    compute_multi_instrength,
    compute_multi_outdegree,
    compute_multi_outstrength,
    get_sparse_trace,
)


@pytest.fixture
def sample_adjacency_matrix():
    """
    Create a large 30x30 sparse COO adjacency matrix for comprehensive testing.
    This represents a multilayer network with n=10 nodes and l=3 layers.
    
    Matrix structure (30x30 for n=10, l=3):
    Node structure: [0_0, 0_1, 0_2, 1_0, 1_1, 1_2, ..., 9_0, 9_1, 9_2] 
    where i_j means node i in layer j
    """
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

    return {
        'matrix': sparse_matrix,
        'n': n,
        'l': l,
        'nnz': sparse_matrix._nnz(),
        'density': sparse_matrix._nnz() / (matrix_size * matrix_size),
        'entries': entries  # Original entries for reference
    }

@pytest.fixture
def expected_structural_analysis():
    """
    Fixture providing expected in/out degrees for sample_adjacency_matrix.
    
    Calculations based on the adjacency matrix structure:
    - Regular degree: sum across all layers for each node
    - Multi-degree: count of inter-layer connections for each node
    """
    return {
        # In-degrees: incoming edges to each node (sum across both layers)
        'in_degrees': torch.tensor([[5,3,5],
                                    [3,4,6],
                                    [4,5,5],
                                    [7,6,2],
                                    [6,4,3],
                                    [6,3,5],
                                    [5,6,3],
                                    [3,4,2],
                                    [4,4,4],
                                    [5,5,1],
                                    ], dtype=torch.float),
        'in_degrees_for_layer_0': torch.tensor([5, 3, 4,7,6,6,5,3,4,5], dtype=torch.float),
        'in_strengths': torch.tensor([[5,3,5],
                                    [3,4,6],
                                    [4,5,5],
                                    [7,6,2],
                                    [6,4,3],
                                    [6,3,5],
                                    [5,6,3],
                                    [3,4,2],
                                    [4,4,4],
                                    [5,5,1],
                                    ], dtype=torch.float),
        
        # Out-degrees: outgoing edges from each node (sum across both layers)  
        'out_degrees': torch.tensor([[5,3,5],
                                    [3,4,6],
                                    [4,5,5],
                                    [7,6,2],
                                    [6,4,3],
                                    [6,3,5],
                                    [5,6,3],
                                    [3,4,2],
                                    [4,4,4],
                                    [5,5,1],
                                    ], dtype=torch.float),
        'out_strengths': torch.tensor([[5,3,5],
                                    [3,4,6],
                                    [4,5,5],
                                    [7,6,2],
                                    [6,4,3],
                                    [6,3,5],
                                    [5,6,3],
                                    [3,4,2],
                                    [4,4,4],
                                    [5,5,1],
                                    ], dtype=torch.float),

        # Multi-indegrees: inter-layer incoming connections
        # This counts connections FROM other layers TO each node
        'multi_in_degrees': torch.tensor([[2,2,2],
                                    [2,2,2],
                                    [2,2,2],
                                    [2,2,2],
                                    [2,2,2],
                                    [2,2,2],
                                    [2,2,2],
                                    [2,2,2],
                                    [2,2,2],
                                    [2,2,2],
                                    ], dtype=torch.float),
        'multi_in_strengths': torch.tensor([[2,2,2],
                                    [2,2,2],
                                    [2,2,2],
                                    [2,2,2],
                                    [2,2,2],
                                    [2,2,2],
                                    [2,2,2],
                                    [2,2,2],
                                    [2,2,2],
                                    [2,2,2],
                                    ], dtype=torch.float),
        # Multi-outdegrees: inter-layer outgoing connections  
        # This counts connections TO other layers FROM each node
        'multi_out_degrees': torch.tensor([[2,2,2],
                                    [2,2,2],
                                    [2,2,2],
                                    [2,2,2],
                                    [2,2,2],
                                    [2,2,2],
                                    [2,2,2],
                                    [2,2,2],
                                    [2,2,2],
                                    [2,2,2],
                                    ], dtype=torch.float),
        'multi_out_strengths': torch.tensor([[2,2,2],
                                    [2,2,2],
                                    [2,2,2],
                                    [2,2,2],
                                    [2,2,2],
                                    [2,2,2],
                                    [2,2,2],
                                    [2,2,2],
                                    [2,2,2],
                                    [2,2,2],
                                    ], dtype=torch.float),

        # Clustering coefficient:
        'global_average_clustering': 0.240458,  # No triangles in this example
    }

@pytest.fixture
def expected_sparse_properties():
    """
    Fixture providing expected properties of the sparse matrix. 
    """

    n, l = 10, 3
    matrix_size = n * l
    trace = 0.0
    return {
        'trace': trace,  # Trace should be zero for this specific example
        'n': n,
        'l': l,
        'matrix_size': matrix_size,
        'nnz': 188,  # Number of non-zero entries in the sparse matrix
        'density': 188 / (matrix_size * matrix_size)  # Density of the sparse matrix
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

def test_compute_indegree_for_layer(sample_adjacency_matrix, expected_structural_analysis):
    """
    Test the computation of indegree for a specific layer from a sparse adjacency matrix.
    """
    layer_index = 0  # Test for the first layer
    indegree = compute_indegree_for_layer(sample_adjacency_matrix["matrix"], 
                                           layer_index=layer_index, 
                                           n=sample_adjacency_matrix['n'], 
                                           l=sample_adjacency_matrix['l'])

    expected = expected_structural_analysis['in_degrees_for_layer_0']
    assert indegree.shape == expected.shape, f"Shape mismatch: got {indegree.shape}, expected {expected.shape}"
    assert torch.allclose(indegree, expected, atol=1e-5), f"In-degree mismatch: got {indegree}, expected {expected}"


def test_compute_indegree(sample_adjacency_matrix, expected_structural_analysis):
    """
    Test the computation of indegree from a sparse adjacency matrix.
    """
    indegree = compute_indegree(sample_adjacency_matrix["matrix"], 
                                n=sample_adjacency_matrix['n'], 
                                l=sample_adjacency_matrix['l'])

    expected = expected_structural_analysis['in_degrees']
    assert indegree.shape == expected.shape, f"Shape mismatch: got {indegree.shape}, expected {expected.shape}"
    assert torch.allclose(indegree, expected, atol=1e-5), f"In-degree mismatch: got {indegree}, expected {expected}"


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



def test_compute_multi_indegree(sample_adjacency_matrix, expected_structural_analysis):
    """
    Test the computation of multi-indegree from a sparse adjacency matrix.
    Multi-indegree counts inter-layer incoming connections.
    """
    multi_indegree = compute_multi_indegree(sample_adjacency_matrix["matrix"], 
                                            n=sample_adjacency_matrix['n'], 
                                            l=sample_adjacency_matrix['l'])

    expected = expected_structural_analysis['multi_in_degrees']
    assert multi_indegree.shape == expected.shape, f"Shape mismatch: got {multi_indegree.shape}, expected {expected.shape}"
    assert torch.allclose(multi_indegree, expected, atol=1e-5), f"Multi-in-degree mismatch: got {multi_indegree}, expected {expected}"


def test_compute_multi_outdegree(sample_adjacency_matrix, expected_structural_analysis):
    """
    Test the computation of multi-outdegree from a sparse adjacency matrix.
    Multi-outdegree counts inter-layer outgoing connections.
    """
    multi_outdegree = compute_multi_outdegree(sample_adjacency_matrix["matrix"], 
                                            n=sample_adjacency_matrix['n'], 
                                            l=sample_adjacency_matrix['l'])

    expected = expected_structural_analysis['multi_out_degrees']
    assert multi_outdegree.shape == expected.shape, f"Shape mismatch: got {multi_outdegree.shape}, expected {expected.shape}"
    assert torch.allclose(multi_outdegree, expected, atol=1e-5), f"Multi-out-degree mismatch: got {multi_outdegree}, expected {expected}"

def test_compute_instrengths(sample_adjacency_matrix, expected_structural_analysis):
    """
    Test the computation of in-strengths from a sparse adjacency matrix.
    """
    instrengths = compute_instrength(sample_adjacency_matrix["matrix"], 
                                    n=sample_adjacency_matrix['n'], 
                                    l=sample_adjacency_matrix['l'])

    expected = expected_structural_analysis['in_strengths']
    assert instrengths.shape == expected.shape, f"Shape mismatch: got {instrengths.shape}, expected {expected.shape}"
    assert torch.allclose(instrengths, expected, atol=1e-5), f"In-strength mismatch: got {instrengths}, expected {expected}"

def test_get_sparse_trace(sample_adjacency_matrix, expected_sparse_properties):
    """
    Test that we can extract the trace of a sparse matrix correctly.
    The trace should be the sum of diagonal elements.
    """
    sparse_matrix = sample_adjacency_matrix['matrix']
    
    # Get the trace
    trace = get_sparse_trace(sparse_matrix)

    expected_trace = expected_sparse_properties['trace']
    assert torch.isclose(torch.tensor(trace), torch.tensor(expected_trace)), f"Expected trace {expected_trace}, got {trace.item()}"

def test_compute_average_global_clustering(sample_adjacency_matrix, expected_structural_analysis):
    """
    Test the computation of average global clustering coefficient.
    This should be zero for the given example since there are no triangles.
    """
    sparse_matrix = sample_adjacency_matrix['matrix']
    n, l = sample_adjacency_matrix['n'], sample_adjacency_matrix['l']

    g = nx.from_numpy_array(sparse_matrix.to_dense().numpy(), create_using=nx.DiGraph)
    expected_clustering_coefficient = nx.average_clustering(g)

    clustering_coefficient = compute_average_global_clustering(sparse_matrix, n=n, l=l)
    print(f"Computed clustering coefficient: {clustering_coefficient} and expected: {expected_clustering_coefficient}")
    assert clustering_coefficient == expected_clustering_coefficient, \
        f"Expected clustering coefficient {expected_clustering_coefficient}, got {clustering_coefficient}"
