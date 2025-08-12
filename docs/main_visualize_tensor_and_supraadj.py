import sys
from pathlib import Path
import argparse
# Add the parent directory to the system path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mlnetst.utils.mlnet_utils import build_supra_adjacency_matrix_from_tensor
from mlnetst.utils.mlnet_metrics_utils import compute_indegree, compute_average_global_clustering

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize tensor and supra-adjacency matrix.")
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="default_experiment",
        help="Name of the experiment for saving results.",
    )
    parser.add_argument(
        "--do_heatmap",
        action="store_true",
        help="Whether to create heatmaps for the supra-adjacency matrix.",
    )
    return parser.parse_args()

def create_memory_efficient_heatmap(matrix, max_size=4000, sample_method='downsample'):
    """
    Create a memory-efficient heatmap for large matrices.
    
    Args:
        matrix: torch tensor to visualize
        max_size: maximum size for visualization (default 2000x2000)
        sample_method: 'downsample', 'random_sample', or 'block_average'
    """
    # Move to CPU and convert to numpy if needed
    if matrix.is_cuda:
        matrix = matrix.cpu()
    
    if matrix.is_sparse:
        matrix = matrix.to_dense()
        
    matrix_np = matrix.numpy()
    original_shape = matrix_np.shape
    
    print(f"Original matrix shape: {original_shape}")
    print(f"Original matrix memory usage: {matrix_np.nbytes / 1024**3:.2f} GB")
    
    if min(original_shape) <= max_size:
        print("Matrix is small enough, using original size")
        return matrix_np, original_shape
    
    if sample_method == 'downsample':
        # Simple downsampling - take every nth element
        step = max(original_shape) // max_size
        sampled = matrix_np[::step, ::step]
        print(f"Downsampled matrix shape: {sampled.shape} (step={step})")
        
    elif sample_method == 'block_average':
        # Average over blocks to preserve overall structure
        step = max(original_shape) // max_size
        h, w = original_shape
        new_h, new_w = h // step, w // step
        
        # Reshape and average
        reshaped = matrix_np[:new_h*step, :new_w*step].reshape(new_h, step, new_w, step)
        sampled = reshaped.mean(axis=(1, 3))
        print(f"Block-averaged matrix shape: {sampled.shape} (block_size={step}x{step})")
        
    elif sample_method == 'random_sample':
        # Random sampling of rows and columns
        np.random.seed(42)  # For reproducibility
        row_indices = np.sort(np.random.choice(original_shape[0], min(max_size, original_shape[0]), replace=False))
        col_indices = np.sort(np.random.choice(original_shape[1], min(max_size, original_shape[1]), replace=False))
        sampled = matrix_np[np.ix_(row_indices, col_indices)]
        print(f"Random sampled matrix shape: {sampled.shape}")
        
    else:
        raise ValueError("sample_method must be 'downsample', 'random_sample', or 'block_average'")
    
    print(f"Sampled matrix memory usage: {sampled.nbytes / 1024**3:.2f} GB")
    
    # Apply log transformation to handle scale issues
    print(f"Matrix value range before log transform: [{sampled.min():.6f}, {sampled.max():.6f}]")
    
    # Add small epsilon to avoid log(0) and handle negative values
    epsilon = 1e-10
    
    # Handle negative values and zeros
    if sampled.min() < 0:
        print("Found negative values, shifting to positive range...")
        sampled = sampled - sampled.min() + epsilon
    else:
        sampled = sampled + epsilon
        
    # Apply log transformation
    log_sampled = np.log10(sampled)
    print(f"Matrix value range after log10 transform: [{log_sampled.min():.6f}, {log_sampled.max():.6f}]")
    
    return log_sampled, original_shape

def main():
    args = parse_args()

    t = torch.load(str(Path(__file__).resolve().parents[1] / "data" / "processed" / f"{args.experiment_name}_mlnet.pth"))
    print("Tensor shape:", t.shape)
    print("Tensor data:", t)
    supra_matrix = build_supra_adjacency_matrix_from_tensor(t)
    print("Supra-adjacency matrix shape:", supra_matrix.shape)
    print("Supra-adjacency matrix data:", supra_matrix)

    if args.do_heatmap:
        dense_supra_matrix = supra_matrix.to_dense() if supra_matrix.is_sparse else supra_matrix
        print("Dense Supra-adjacency matrix shape:", dense_supra_matrix.shape)
        print("Dense Supra-adjacency matrix data:", dense_supra_matrix)

        # Create memory-efficient visualization
        print("Creating memory-efficient heatmap...")
        
        # Try different sampling methods
        methods = ['block_average']
        
        for i, method in enumerate(methods):
            try:
                print(f"\n--- Using {method} method ---")
                sampled_matrix, original_shape = create_memory_efficient_heatmap(
                    dense_supra_matrix, 
                    max_size=2000, 
                    sample_method=method
                )
                
                # Create the heatmap
                plt.figure(figsize=(12, 10))
                
                # Use a more memory-efficient plotting approach
                if sampled_matrix.size > 4000000:  # If still > 2000x2000
                    # Further reduce for visualization
                    step = int(np.sqrt(sampled_matrix.size / 1000000))  # Target ~1M elements
                    sampled_matrix = sampled_matrix[::step, ::step]
                    print(f"Further reduced for plotting: {sampled_matrix.shape}")
                
                # Create heatmap with optimized settings
                im = plt.imshow(sampled_matrix, 
                            cmap='viridis', 
                            aspect='auto',
                            interpolation='nearest')
                
                cbar = plt.colorbar(im, shrink=0.8)
                cbar.set_label('log10(values)', rotation=270, labelpad=15)
                plt.title(f"Supra-adjacency matrix heatmap (log10 scale) for {args.experiment_name}\n"
                        f"Original: {original_shape}, Sampled: {sampled_matrix.shape} ({method})")
                plt.xlabel("Nodes and Layers")
                plt.ylabel("Nodes and Layers")
                
                # Save with method name
                output_path = Path(__file__).resolve().parents[1] / "media" / f"{args.experiment_name}_supra_adjacency_{method}_log_heatmap.png"
                plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
                plt.close()  # Important: close to free memory
                
                print(f"Saved heatmap using {method} method to: {output_path}")
                
                # Clean up
                del sampled_matrix
                
            except Exception as e:
                print(f"Error with {method} method: {e}")
                continue
        
        print("\nHeatmap visualization completed!")

    # Let's visualize the indegrees for each layer
    n = t.shape[0]  # Number of nodes
    l = t.shape[1]  # Number of layers
    print(f"Number of nodes: {n}, Number of layers: {l}")
    
    # Compute indegrees
    print("Computing indegrees...")
    indegrees = compute_indegree(supra_matrix, n, l)
    # Indegrees is a tensor of shape (n, l) where each element is the indegree of the corresponding node in the layer
    print("Indegrees shape:", indegrees.shape)
    print("Indegrees data:", indegrees)
    
    # Visualize indegrees
    plt.figure(figsize=(12, 6))
    sns.heatmap(indegrees.numpy(), cmap='coolwarm', cbar=True, annot=False)
    plt.title(f"Indegrees heatmap for {args.experiment_name}")
    plt.xlabel("Layers")
    plt.ylabel("Nodes")
    
    # Save indegrees heatmap
    indegrees_output_path = Path(__file__).resolve().parents[1] / "media" / f"{args.experiment_name}_indegrees_heatmap.png"
    plt.savefig(str(indegrees_output_path), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved indegrees heatmap to: {indegrees_output_path}")

    # Compute global average clustering coefficient
    print("Computing global average clustering coefficient...")
    # Assuming you have a function to compute this, e.g., compute_global_clustering_coefficient(supra_matrix)
    # global_clustering_coefficient = compute_global_clustering_coefficient(supra_matrix)
    # print("Global average clustering coefficient:", global_clustering_coefficient)
    global_clustering_coefficient = compute_average_global_clustering(supra_matrix, n, l)
    print("Global average clustering coefficient:", global_clustering_coefficient)

if __name__ == "__main__":
    main()