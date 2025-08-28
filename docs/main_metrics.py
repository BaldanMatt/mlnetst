import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import torch
import json
import anndata
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from mlnetst.utils.mlnet_metrics_utils import (
    compute_indegree,
    compute_instrength,
    compute_average_global_clustering,
    compute_katz_centrality,
)
from mlnetst.utils.mlnet_utils import build_supra_adjacency_matrix_from_tensor

def parse_args():
    parser = argparse.ArgumentParser(description="Compute network metrics from supra-adjacency matrix.")
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="default_experiment",
        help="Name of the experiment for saving results.",
    )
    return parser.parse_args()

def main(args):
    print(f"Experiment Name: {args.experiment_name}")
    tensor = torch.load(
        Path(__file__).resolve().parents[1] / "data" / "processed" / f"{args.experiment_name}_mlnet.pth"
    )
    node_metadata = anndata.read_h5ad(
        Path(__file__).resolve().parents[1] / "data" / "processed" / f"{args.experiment_name}_subdata.h5ad"
    ).obs
    layer_mapping = json.load(
        open(
            Path(__file__).resolve().parents[1]
            / "data"
            / "processed"
            / f"{args.experiment_name}_layer_mapping.json",
            "r",
        )
    )
    print(tensor)
    print("################################")
    print(node_metadata.info())
    print("################################")
    print(layer_mapping.keys(), "for num of layers:", len(layer_mapping))
    print("################################")

    print("Everything loaded successfully.")
    supra_adjacency_matrix = build_supra_adjacency_matrix_from_tensor(tensor)
    try:
        df_to_plot = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "processed" / f"{args.experiment_name}_katz_centrality.csv", index_col=0)
        print("Katz centrality data loaded from file.")
    except FileNotFoundError:
        print("Global clustering coefficient")
        start_time = time()
        num_nodes = tensor.size(0)
        num_layers = tensor.size(1)
        gc = compute_average_global_clustering(supra_adjacency_matrix,n=num_nodes,l=num_layers)
        end_time = time()
        print(f"Global clustering coefficient: {gc:.4f} in {end_time - start_time:.2f} seconds")
        print("Compute katz centrality for each node")
        start_time = time()
        katz = compute_katz_centrality(supra_adjacency_matrix,n=num_nodes,l=num_layers)
        end_time = time()
        print(f"Katz centrality computed in {end_time - start_time:.2f} seconds")


        df_to_plot = pd.DataFrame({
            "katz": katz,
            "centroid_x": np.asarray(node_metadata['centroid_x']),
            "centroid_y": np.asarray(node_metadata['centroid_y']),
        }, index=node_metadata.index)
        # save to file
        df_to_plot.to_csv(Path(__file__).resolve().parents[1] / "data" / "processed" / f"{args.experiment_name}_katz_centrality.csv")
    # Clean, easy-on-the-eye spatial scatter of Katz centrality
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(
        df_to_plot['centroid_x'],
        df_to_plot['centroid_y'],
        c=df_to_plot['katz'],
        cmap='viridis',
        s=30,
        alpha=0.85,
        linewidths=0
    )
    ax.set_title('Katz centrality (spatial distribution)', fontsize=14)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_aspect('equal', adjustable='datalim')

    # Minimalist styling
    sns.despine(ax=ax, trim=True)
    ax.tick_params(axis='both', which='both', length=0, labelsize=10)
    ax.grid(False)

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label('Katz centrality', fontsize=11)

    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().parents[1] / "media" / f"{args.experiment_name}_katz_centrality.png", dpi=150)
    plt.close()
    print("Katz centrality plot saved.")

if __name__ == "__main__":
    args = parse_args()
    main(args)