import sys
from pathlib import Path
import os
sys.path.append(str(Path(__file__).parents[1]))
import pandas as pd
import anndata as ad
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import random
from mlnetst.core.knowledge.networks import load_resource
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
pd.core.common.random_state(None)
random.seed(RANDOM_SEED)

x_hat_s = ad.read_h5ad(Path(__file__).parents[1] / "data" / "processed" / "mouse1_slice153_x_hat_s.h5ad")
print(x_hat_s)

source, target = "Astro", "L2/3 IT"
subdata = x_hat_s[x_hat_s.obs["subclass"].isin([source,target]), :]
print(subdata)

notFound = True
L = 5
lr_interactions_df = load_resource("mouseconsensus")
# Convert subdata.var_names to lowercase set for faster lookup
valid_genes = set(g.lower() for g in subdata.var_names)

# Filter interactions where both source and target (first letters or elements) are in valid_genes
filtered_df = lr_interactions_df[
    lr_interactions_df["source"].apply(lambda s: list(s)[0].lower() in valid_genes) &
    lr_interactions_df["target"].apply(lambda t: list(t)[0].lower() in valid_genes)
]

# Sample from filtered interactions
if len(filtered_df) >= L:
    sample_lr = filtered_df.sample(n=L)
    print(sample_lr)
else:
    print(f"Not enough valid interactions (found {len(filtered_df)}, needed {L})")
    

id_lr = 0
MODE = 0
TOLL_DIST = 1e-5
TOLL_GEOM_MEAN = 1e-10
TH_INTERCELLULAR = 0.99

def compute_intra_layer_score(source_expression, # N x 1
                              target_expression, # N x 1 -> 1 x N
                              distance_matrix,
                             toll_dist): # -> N x N 
                
    single_lr_scores = torch.div(source_expression @ target_expression.permute(*torch.arange(target_expression.ndim - 1, -1, -1)),
                                 distance_matrix + toll_dist)
    single_lr_scores = single_lr_scores.fill_diagonal_(0)

    return single_lr_scores

def compute_inter_layer_score(source_expression, # L x N
                              target_expression, # L x N -> 1 x L x N 
                              ): # -> L x L x N
    single_inter_score = source_expression @ target_expression.permute(*torch.arange(target_expression.ndim - 1, -1, -1))
    single_inter_score = single_inter_score.fill_diagonal_(0)
    return single_inter_score

def compute_dense_mlnet():
    nlayers = sample_lr.shape[0]
    ncells = subdata.shape[0]

    # Init big tensor to hold the mlnet
    mlnet = torch.zeros((ncells, nlayers, ncells, nlayers), dtype=torch.float32)

    # Prepare spatial positions
    subdata.obsm["spatial"] = np.array([(x,y) for x,y in zip(subdata.obs["centroid_x"], subdata.obs["centroid_y"])])
    # Cast spatial coordinates to tensor Ncells x 2
    spatial_position_tensor = torch.tensor(
        subdata.obsm["spatial"], dtype=torch.float32
    )
    # Compute euclidean distance between points in space
    dist_matrix = torch.cdist(
        spatial_position_tensor,
        spatial_position_tensor,
        p=2
    )

    for ilayer in range(nlayers):
        ligand = list(sample_lr["source"].explode())[ilayer].lower()
        receptor = list(sample_lr["target"].explode())[ilayer].lower()
        print(f"Computing the following layer: {ligand} - {receptor}")
        
        # Select ligand expression   
        ligand_expression = torch.from_numpy(
            subdata[:, list(sample_lr["source"].explode())[ilayer].lower().split("_")].X.astype(np.float32)
        ).squeeze()
        if len(ligand.split("_"))>1:
            print(f"\tligand {ligand} is a complex")
            ligand_expression = torch.exp(torch.mean(torch.log(ligand_expression + toll_geom_mean), dim=1))
        else:
            print(f"\tligand {ligand} is not a complex")
        
        # Select receptor expression  
        receptor_expression = torch.from_numpy(
            subdata[:, list(sample_lr["target"].explode())[ilayer].lower().split("_")].X.astype(np.float32)
        ).squeeze()
        if len(receptor.split("_"))>1:
            print(f"\treceptor {receptor} is a complex")
            receptor_expression = torch.exp(torch.mean(torch.log(receptor_expression + toll_geom_mean), dim=1))
        else:
            print(f"\treceptor {receptor} is not a complex")
    
        
def compute_layers(subdata,
                        sample_lr,
                        mode: int = 1,
                        toll_dist: float = TOLL_DIST,
                        toll_geom_mean: float = TOLL_GEOM_MEAN,
                        th_per_layer: float = TH_INTERCELLULAR):
    # prepare dimensions
    num_cells = subdata.shape[0]
    num_layers = sample_lr.shape[0]

    # Prepare spatial positions
    subdata.obsm["spatial"] = np.array([(x,y) for x,y in zip(subdata.obs["centroid_x"], subdata.obs["centroid_y"])])
    # Cast spatial coordinates to tensor Ncells x 2
    spatial_position_tensor = torch.tensor(
        subdata.obsm["spatial"], dtype=torch.float32
    )
    # Compute euclidean distance between points in space
    dist_matrix = torch.cdist(
        spatial_position_tensor,
        spatial_position_tensor,
        p=2
    )
    torch_geom = lambda t, *a, **kw: t.log().mean(*a, **kw).exp()
    
    # Init loop for layers
    for ilayer in range(num_layers):
        ligand = list(sample_lr["source"].explode())[ilayer].lower()
        receptor = list(sample_lr["target"].explode())[ilayer].lower()
        print(f"Computing the following layer: {ligand} - {receptor}")
        
        # Select ligand expression   
        ligand_expression = torch.from_numpy(
            subdata[:, list(sample_lr["source"].explode())[ilayer].lower().split("_")].X.astype(np.float32)
        ).squeeze()
        if len(ligand.split("_"))>1:
            print(f"\tligand {ligand} is a complex")
            ligand_expression = torch.exp(torch.mean(torch.log(ligand_expression + toll_geom_mean), dim=1))
        else:
            print(f"\tligand {ligand} is not a complex")
        
        # Select receptor expression  
        receptor_expression = torch.from_numpy(
            subdata[:, list(sample_lr["target"].explode())[ilayer].lower().split("_")].X.astype(np.float32)
        ).squeeze()
        if len(receptor.split("_"))>1:
            print(f"\treceptor {receptor} is a complex")
            receptor_expression = torch.exp(torch.mean(torch.log(receptor_expression + toll_geom_mean), dim=1))
        else:
            print(f"\treceptor {receptor} is not a complex")
    
        single_lr_scores = compute_intra_layer_score(
            source_expression=ligand_expression,
            target_expression=receptor_expression,
            distance_matrix=dist_matrix,toll_dist=toll_dist
        )
    
        # Evaluate mode to do sparsity
        if mode == 1:
            # Threshold naively each layer
            th_mask = single_lr_scores > np.quantile(single_lr_scores.numpy().flatten(), th_per_layer)
            single_lr_scores = single_lr_scores * th_mask
            # Init variable mlnet
            single_lr_scores = single_lr_scores.to_sparse(layout=torch.sparse_coo)
            if ilayer == 0:
                mlnet = single_lr_scores.unsqueeze(2)
            else:
                mlnet = torch.cat((mlnet, single_lr_scores.unsqueeze(2)), axis=2)
        elif mode == 2:
            # Eventually we could threshold the whole set of intralayer interactions to keep only the most significant. Whatever significant means... We would need to make the differen intralayer scores comparable between each other.
            pass
        elif mode == 3:
            # Eventually we could aggregate nodes to reduce the amount of intralayer interactions...
            pass
        elif mode == 0:
            # We are not sparsing the tensor
            if ilayer == 0:
                mlnet = single_lr_scores.unsqueeze(2)
            else:
                mlnet = torch.cat((mlnet, single_lr_scores.unsqueeze(2)), axis = 2)
    
    for icell in range(num_cells):
        # Select Receptor expression of every layer

        receptor_list = list(sample_lr["target"])
        ligand_list = list(sample_lr["source"])
        print(ligand_list, receptor_list)
        ligand_expression = torch.zeros((len(ligand_list),1), dtype = torch.float32)
        print(f"Shape of the ligand expression tensor: {ligand_expression.shape}")
        for iligand, ligand in enumerate(ligand_list):
            ligand = ligand.lower()
            if len(ligand.split("_"))>1:
                print(f"\tligand {ligand} is a complex")
                ligand_expression[iligand] = torch.exp(torch.mean(torch.log(
                    subdata[icell, ligand.split("_")].X.astype(np.float32) + toll_geom_mean), dim=1))
            else:
                print(f"\tligand {ligand} is not a complex")
                ligand_expression[iligand] = torch.from_numpy(
                    subdata[icell, ligand].X.astype(np.float32)
                ).squeeze()
        receptor_expression = torch.zeros((len(receptor_list),1), dtype = torch.float32)
        print(f"Shape of the receptor expression tensor: {receptor_expression.shape}")
        for ireceptor, receptor in enumerate(receptor_list):
            receptor = receptor.lower()
            if len(receptor.split("_"))>1:
                print(f"\treceptor {receptor} is a complex")
                receptor_expression[ireceptor] = torch.exp(torch.mean(torch.log(
                    subdata[icell, receptor.split("_")].X.astype(np.float32) + toll_geom_mean), dim=1))
            else:
                print(f"\treceptor {receptor} is not a complex")
                receptor_expression[ireceptor] = torch.from_numpy(
                    subdata[icell, receptor].X.astype(np.float32)
                ).squeeze()
      
        cell_interlayer_scores = compute_inter_layer_score(
            source_expression=ligand_expression,
            target_expression=receptor_expression
        )
        print(f"Shape of the interlayer scores: {cell_interlayer_scores.shape}")
        if icell == 0:
            mlnet = torch.cat((mlnet.unsqueeze(3), cell_interlayer_scores.unsqueeze(0).unsqueeze(0)), axis=3)
        else:
            mlnet = torch.cat((mlnet, cell_interlayer_scores.unsqueeze(0).unsqueeze(0)), axis=3)
        # Select Ligand expression of every layer
        if icell >= 7:
            break

    return mlnet


    

mlnet = compute_layers(
    subdata=subdata,
    sample_lr=sample_lr,
    mode=MODE,
    toll_dist=TOLL_DIST,th_per_layer=TH_INTERCELLULAR
                      )
if MODE == 0:
    typeoftensor = "dense"
else:
    typeoftensor = "sparse"
    mlnet = mlnet.coalesce()

#for ilayer in range(mlnet.shape[2]):
#     Let's display flatten the layer and display the distribution of values
    #if MODE != 0:
        #layer_values = mlnet[:,:,ilayer].values().cpu().numpy()
    #else:
        #layer_values = mlnet[:,:,ilayer].numpy().flatten()
    #plt.figure(figsize=(8, 4))
    #plt.hist(layer_values, bins=50, alpha=0.7, color='blue')
    #plt.title(f"Distribution of values in layer {ilayer}")
    #plt.xlabel("Value")
    #plt.ylabel("Frequency")
    #plt.grid()
    #plt.show()
#


if MODE != 0:
    import seaborn as sns
    # Get the coalesced version of the sparse tensor (merges duplicates)
    layer_to_plot = 0
    fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize=(12,12))
    axs = axs.flatten()
    layers_to_plot = range(len(axs)) if len(axs) < L else range(L)

    N = mlnet.shape[0]
    subdata.obsm["spatial"] = np.array([(x,y) for x,y in zip(subdata.obs["centroid_x"], subdata.obs["centroid_y"])])
        # Cast spatial coordinates to tensor Ncells x 2
    spatial_position_tensor = torch.tensor(
        subdata.obsm["spatial"], dtype=torch.float32
    )

    indices = mlnet.indices()  # shape: (3, nnz)
    values = mlnet.values()    # shape: (nnz, ...)
    for layer_to_plot in layers_to_plot:
        print(f"Building layer {layer_to_plot}")
        # Get mask for indices of specific layer
        mask = indices[2] == layer_to_plot
        filtered_indices = indices[:, mask]
        filtered_values = values[mask]

        # Project to 2D (drop third dimension)
        row = filtered_indices[0].cpu().numpy()
        col = filtered_indices[1].cpu().numpy()
        val = filtered_values.cpu().numpy()

        # Step 5: Build dense NumPy matrix
        dense_matrix = np.zeros((N, N), dtype=val.dtype)
        dense_matrix[row, col] = val

        # If this is an adjacency matrix (undirected or directed)
        G = nx.from_numpy_array(dense_matrix, create_using=nx.DiGraph())  # undirected

        # Visualize network
        print(f"\tPreparing layer viz")
        pos = [(x,y) for x,y in zip(spatial_position_tensor[:,0].numpy(), spatial_position_tensor[:,1].numpy())]


        # --- Node color: categorical subclass from AnnData.obs ---
        subclasses = subdata.obs['subclass']  # pandas Series of shape (n_nodes,)

        # Map categories to colors using seaborn color palette
        unique_classes = subclasses.unique()
        palette = sns.color_palette("hls", len(unique_classes))
        class2color = {cls: palette[i] for i, cls in enumerate(unique_classes)}

        node_color = [class2color[subclasses.iloc[i]] for i in range(len(G.nodes))]

        # --- Node size: in-degree ---
        node_size = [1 + 0.5 * G.in_degree(i) for i in range(len(G.nodes))]  # scale as needed

        # --- Edge transparency: based on edge weights ---
        # Get edge weights for alpha scaling
        edge_weights = np.array([G[u][v]['weight'] for u, v in G.edges])
        if len(edge_weights) == 0:
            edge_weights = np.array([1])
        norm_weights = edge_weights / edge_weights.max()
        edge_width = 0.1 + 0.25 * norm_weights  # make visible but not too thick
        print(f"\tEdges...")
        # Draw edges with individual alpha per edge
        nx.draw_networkx_edges(
            G, pos,
            width=edge_width,
            edge_color='gray',
            ax = axs[layer_to_plot]
        )

        # Draw nodes
        print(f"\tNodes...")
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_color,
            node_size=node_size,
            ax = axs[layer_to_plot]
        )

        axs[layer_to_plot].set_title(f"Ligand {sample_lr["source"].explode().iloc[layer_to_plot].lower()} - Receptor {sample_lr["target"].explode().iloc[layer_to_plot].lower()}")

    plt.show()


