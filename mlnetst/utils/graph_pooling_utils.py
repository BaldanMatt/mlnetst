"""
In this module I would like to define some utility functions for pooling graphs.

It will contain these following approaches:
 - Spatial pooling by aggregating with some function the features of nodes that have the same or not label close to each other in space
 - Pooling using some graph reduction methods that were introduced in the tutorial at IJCNN2025
"""

from pathlib import Path
import os
import anndata
import matplotlib.pyplot as plt
import torch

if __name__ == "__main__":
    print("DEBUGGING graph_pooling_utils.py")
    # Here you can add some test cases or examples of how to use the functions defined in this module.
    # For example, you might want to create a sample graph and apply pooling functions to it.
    pass  # Replace with actual test code if needed.

    x = anndata.read_h5ad(Path(__file__).parents[2] / "data" / "processed" / "mouse1_slice153_x_f.h5ad")
    print(x)
    print(f"{x.obs["centroid_x"].min():.2f}, {x.obs["centroid_x"].max():.2f}\n" + \
          f"{x.obs["centroid_y"].min():.2f}, {x.obs["centroid_y"].max():.2f}")

    offset_factor = 0.05  # Offset as 10% of the range
    x_range = x.obs["centroid_x"].max() - x.obs["centroid_x"].min()
    y_range = x.obs["centroid_y"].max() - x.obs["centroid_y"].min()
    offset_x = offset_factor * x_range
    offset_y = offset_factor * y_range

    m_x = x.obs["centroid_x"].min() - offset_x
    M_x = x.obs["centroid_x"].max() + offset_x
    m_y = x.obs["centroid_y"].min() - offset_y
    M_y = x.obs["centroid_y"].max() + offset_y

    print(f"X range: {m_x:.2f} to {M_x:.2f}")
    print(f"Y range: {m_y:.2f} to {M_y:.2f}")   
    n_rows_grid = 100
    n_cols_grid = 100
    grid_x = torch.linspace(m_x, M_x, n_cols_grid)
    grid_y = torch.linspace(m_y, M_y, n_rows_grid)

    # Aggregate features within the grid
    cell_grid_x = torch.bucketize(torch.Tensor(x.obs["centroid_x"]), grid_x)
    cell_grid_y = torch.bucketize(torch.Tensor(x.obs["centroid_y"]), grid_y)
    print(f"Cell grid X: {cell_grid_x}")
    print(f"Cell grid Y: {cell_grid_y}")


    print(f"Grid X: {grid_x}")
    fig, ax = plt.subplots(figsize=(10, 10))    
    ax.set_xlim(m_x, M_x)
    ax.set_ylim(m_y, M_y)
    ax.set_xticks(grid_x)
    ax.set_yticks(grid_y)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    # Color the grid pathces based on how many cells are within each cell
    for i in range(n_rows_grid):
        for j in range(n_cols_grid):
            if i < n_rows_grid - 1 and j < n_cols_grid - 1:  # Avoid out-of-bounds for the last row/column
                cell_mask = (cell_grid_x -1 == j) & (cell_grid_y -1 == i)
                cell_count = cell_mask.sum().item()
                if cell_count > 0:
                    # Color the rectangle based on the number of cells
                    color_intensity = min(cell_count / 10, 1.0)
                    color = (1.0 - color_intensity, 1.0, 1.0)
                    # use viridis cmap
                    color = plt.cm.viridis(color_intensity)

                    ax.add_patch(plt.Rectangle((grid_x[j], grid_y[i]), 
                                            grid_x[j+1] - grid_x[j], 
                                            grid_y[i+1] - grid_y[i], 
                                            color=color, alpha=0.5))
    ax.set_title("Grid for Spatial Pooling")
    ax.set_xlabel("Centroid X")
    ax.set_ylabel("Centroid Y")
    ax.scatter(x.obs["centroid_x"], x.obs["centroid_y"], s=10, c='blue', label='Nodes')
    plt.show()


