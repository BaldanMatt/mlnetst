import mlnetst as mnst
import anndata as ad
import pathlib, os
from sklearn.metrics import mutual_info_score, pairwise_distances
import numpy as np
import numba as nb
from pandas import DataFrame, get_dummies
import pandas as pd
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import squidpy as sq
import warnings
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import random
import math

#from feature_engine.encoding import RareLabelEncoder, OrdinalEncoder
RANDOM_SEED = 42
N_FEATURES_TO_KEEP = None
K_NEIGHBORS = 5
np.random.seed(RANDOM_SEED)

warnings.filterwarnings("ignore")


def find_k_nearest_neighbors(distance_matrix, k):
    """Find k-nearest neighbors for each observation based on distance matrix."""
    n_samples = distance_matrix.shape[0]
    # Get indices of k+1 smallest distances (including self)
    nearest_neighbors_idx = np.argsort(distance_matrix, axis=1)[:, 1:k + 1]
    # Get the corresponding distances
    nearest_neighbors_dist = np.take_along_axis(distance_matrix, nearest_neighbors_idx, axis=1)
    return nearest_neighbors_idx, nearest_neighbors_dist

def main():
    print("Hello from mlnetst!")
    current_working_directory = pathlib.Path().absolute()
    data_directory = current_working_directory / "data"
    # Distinguish raw and processed data
    raw_data_dir = data_directory / "raw"
    processed_data_dir = data_directory / "processed"
    media_dir = data_directory / "media"

    # Loading DATA
    x_f = ad.read_h5ad(processed_data_dir / "mouse1_slice153_x_f.h5ad")
    print(x_f)

    # Subsample features to simplify development
    if N_FEATURES_TO_KEEP is not None:
        selected_features = np.random.choice(x_f.var_names, size=N_FEATURES_TO_KEEP, replace=False)
        x_f = x_f[:, selected_features]

    # extract the mutual information of each TF
    try:
        mi_matrix = pd.read_csv(processed_data_dir / "mouse1_slice153_mi_matrix.csv", index_col=0)
    except FileNotFoundError:
        mi_matrix = pd.DataFrame(
            np.zeros((len(x_f.var_names), len(x_f.var_names))),
            columns = x_f.var_names,
            index = x_f.var_names
        )
        print(type(x_f[:,0].X.flatten()), x_f[:,0].X.flatten().shape)
        for i, tf_i in enumerate(tqdm(x_f.var_names, desc="Outer loop...")):
            for j, tf_j in enumerate(tqdm(x_f.var_names, desc="Inner loop...", leave=False)):
                if i == j:
                    continue
                mi_matrix.iloc[i,j] = mutual_info_regression(x_f[:,tf_i].X.reshape(-1,1),
                                                             x_f[:,tf_j].X.flatten(),
                                                             n_jobs = None)
        print()
        # Remove diagonal
        mi_matrix.to_csv(processed_data_dir / "mouse1_slice153_mi_matrix.csv")

    # Extract the euclidean distance matrix between cells
    row_hue_feature = "class_label"
    unique_categories = x_f.obs[row_hue_feature].unique()
    hue_feature_map = dict(zip(unique_categories, sns.color_palette("tab10", len(unique_categories)).as_hex()))
    row_colors = pd.DataFrame(
        [hue_feature_map[cat] for cat in x_f.obs[row_hue_feature]],
        index=x_f.obs_names
    )
    d_matrix = pairwise_distances(x_f.X, metric="euclidean")

    # Find K nearest neighbors for each observation
    neighbor_indices, neighbor_distances = find_k_nearest_neighbors(d_matrix, K_NEIGHBORS)
    print(f"\nFirst observation's {K_NEIGHBORS} nearest neighbors:")
    print(f"Indices: {neighbor_indices[0]}")
    print(f"Distances: {neighbor_distances[0]}")

    # Create graph
    g = nx.Graph()
    # Add nodes and edges based on nearest neighbors
    for i in tqdm(range(len(x_f.obs_names)), desc="Outer loop..."):
        g.add_node(i, color=hue_feature_map[x_f.obs[row_hue_feature].iloc[i]],
                   x=x_f.obs["centroid_x"].iloc[i],
                   y=x_f.obs["centroid_y"].iloc[i],
                   label=x_f.obs[row_hue_feature].iloc[i])
        for neighbor_idx in tqdm(neighbor_indices[i], desc="Inner loop...", leave=False):
            g.add_edge(i, neighbor_idx)
    print()

    nx.write_graphml(g, media_dir / "mouse1_slice153_graph.graphml")

    # Find the spatial neighbors
    sq.gr.spatial_neighbors(x_f, coord_type = "generic", n_neighs = K_NEIGHBORS)

    # Create an adjacency matrix from neighbor_indices
    adj_matrix = np.zeros((len(x_f.obs_names), len(x_f.obs_names)))
    for i in tqdm(range(len(x_f.obs_names)), desc="Outer loop..."):
        for neighbor_idx in tqdm(neighbor_indices[i], desc="Inner loop...", leave=False):
            adj_matrix[i, neighbor_idx] = 1
    print(adj_matrix.shape, x_f.obsp["spatial_connectivities"].shape)
    sp_adj_matrix = adj_matrix * x_f.obsp["spatial_connectivities"]
    print(np.sum(adj_matrix!=0)/(adj_matrix.shape[0]*adj_matrix.shape[1]),
          "\n",
          np.sum(sp_adj_matrix!=0)/(sp_adj_matrix.shape[0]*sp_adj_matrix.shape[1]))

    # print(sp_adj_matrix.shape, np.sum(sp_adj_matrix!=0)/(sp_adj_matrix.shape[0]*sp_adj_matrix.shape[1]))
    #
    # g = nx.from_numpy_array(sp_adj_matrix)
    # print(g)


    #nx.write_graphml(g, media_dir / "mouse1_slice153_spatial_graph.graphml")

if __name__ == "__main__":
    main()
