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

from mlnetst.utils.mlnet_metrics_utils import (
    compute_indegree,
    compute_instrength,
    compute_average_global_clustering,
    compute_katz_centrality,
)

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
    print("Global clustering coefficient")
    start_time = time()
    num_nodes = tensor.size(0)
    num_layers = tensor.size(1)
    gc = compute_average_global_clustering(tensor,n=num_nodes,l=num_layers)
    end_time = time()
    print(f"Global clustering coefficient: {gc:.4f} in {end_time - start_time:.2f} seconds")
    print("Compute katz centrality for each node")
    start_time = time()
    katz = compute_katz_centrality(tensor,n=num_nodes,l=num_layers)
    end_time = time()
    print(f"Katz centrality computed in {end_time - start_time:.2f} seconds")
    
    fig = plt.figure(figsize=(10, 6))
    node_tensor_indices = tensor.indices()[0]
    df_to_plot = pd.DataFrame({
        'obs_name': node_metadata.index,
        'tensor_idx': node_tensor_indices,
        'x': node_metadata.loc[:, "centroid_x"],
        'y': node_metadata.loc[:, "centroid_y"],
        'katz': katz
    })
    sns.scatterplot(data=df_to_plot, x='x', y='y', hue='katz', palette='coolwarm', legend='full')
    plt.title('Katz Centrality Visualization')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.colorbar(label='Katz Centrality')
    plt.savefig(Path(__file__).resolve().parents[1] / "media" / f"{args.experiment_name}_katz_centrality.png")
    plt.close()
    print("Katz centrality plot saved.")

if __name__ == "__main__":
    args = parse_args()
    main(args)