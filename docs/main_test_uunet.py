import uunet.multinet as ml

import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlnetst.utils.mlnet_utils import build_edgelist_from_tensor
import torch
import anndata
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Test UUNet on a given dataset.")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment to run.")
    parser.add_argument("--do_plot", action="store_true", help="Whether to plot the results.")
    parser.add_argument("--do_community_detection", action="store_true", help="Whether to run community detection methods.")
    return parser.parse_args()

def main():
    args = parse_args()
    experiment_name = args.experiment_name

    # Load the tensor
    t = torch.load(str(Path(__file__).resolve().parents[1] / "data" / "processed" / f"{experiment_name}_mlnet.pth"))
    edge_list = build_edgelist_from_tensor(t)
    print(edge_list.info())
    # load layer mapping
    layer_mapping = pd.read_json(str(Path(__file__).resolve().parents[1] / "data" / "processed" / f"{experiment_name}_layer_mapping.json"))
    print(layer_mapping)
    # load subdata
    adata = anndata.read_h5ad(str(Path(__file__).resolve().parents[1] / "data" / "processed" / f"{experiment_name}_subdata.h5ad"))
    print(adata)

   
    if args.do_community_detection:
        mlnet = ml.empty(experiment_name)

        node_names = adata.obs_names.tolist()
        layer_names = layer_mapping.loc["layer_name",:].tolist()
        print(f"Found {len(node_names)} unique nodes and {len(layer_names)} layers")
        print(f"Node names: {node_names[:10]}...")  # Show first 10 nodes
        print(f"Layer names: {layer_names}")
        
        print(mlnet)
        ml.add_layers(mlnet, layer_names, directed=[True] * len(layer_names))
        print(mlnet)
        ml.add_actors(mlnet, node_names)
        print(mlnet)
        
        # Create vertices where each node is replicated across all layers
        vertices_actors = []
        vertices_layers = []
        for node in node_names:
            for layer in layer_names:
                vertices_actors.append(node)
                vertices_layers.append(layer)
        
        vertices_dict = {
            'actor': vertices_actors,
            'layer': vertices_layers
        }
        
        print(f"Created {len(vertices_actors)} vertices (nodes × layers)")
        ml.add_vertices(mlnet, vertices_dict)
        print(mlnet) 

        # Create edges from the edge list
        # Map node indices to actual node names
        from_actor = [node_names[i] for i in edge_list.loc[:, "node.from"].tolist()]
        from_layer = [layer_mapping[x]["layer_name"] for x in edge_list.loc[:, "layer.from"].tolist()]
        to_actor = [node_names[i] for i in edge_list.loc[:, "node.to"].tolist()]
        to_layer = [layer_mapping[x]["layer_name"] for x in edge_list.loc[:, "layer.to"].tolist()]
        edges_dict = {
            'from_actor': from_actor,
            'from_layer': from_layer,
            'to_actor': to_actor,
            'to_layer': to_layer
        }
        ml.add_edges(mlnet, edges = edges_dict)
        comm_dict = dict()
        print("Starting community detection methods...")
        #comm_dict["abacus"] = ml.abacus(mlnet, 4, 2)
        #print("Finished abacus community detection.")
        #comm_dict["cpm"] = ml.clique_percolation(mlnet, 4, 2) # looong
        #print("Finished clique percolation method.")
        comm_dict["glouvain"] = ml.glouvain(mlnet, 0.01, 1)
        print("Finished Louvain community detection.")
        # comm_dict["infomap"] = ml.infomap(mlnet, True, True, False)  # fast
        # print("Finished Infomap community detection.")
        #comm_dict["mdlp"] = ml.mdlp(mlnet) # loong
        #print("Finished MDLP community detection.")
        # comm_dict["flat_ec"] = ml.flat_ec(mlnet) # fast
        # print("Finished flat edge clustering community detection.")
        # comm_dict["flat_nw"] = ml.flat_nw(mlnet) # fast
        # print("Finished flat node weighting community detection.")
        print("Finished community detection methods.")
        # transforms the typical output of the library (dictionaries) into pandas dataframes 
        def df(d):
            return pd.DataFrame.from_dict(d)

        comm = dict()
        for c in comm_dict.keys():
            comm[c] = df(comm_dict[c])

        stats = dict()
        stats["method"] = list(comm.keys())  # Convert dict_keys to list
        stats["num_comm"] = []
        stats["avg_actors_per_comm"] = []
        stats["avg_layers_per_comm"] = []
        stats["perc_clustered_vertices"] = []
        stats["overlapping"] = []

        for method in stats["method"]:
            stats["num_comm"].append( comm[method].cid.nunique() )
            stats["avg_actors_per_comm"].append( comm[method].groupby("cid").nunique().actor.mean() )
            stats["avg_layers_per_comm"].append( comm[method].groupby("cid").nunique().layer.mean() )
            stats["perc_clustered_vertices"].append( comm[method][["actor","layer"]].drop_duplicates().shape[0] / ml.num_vertices(mlnet) )
            stats["overlapping"].append( comm[method].shape[0] / comm[method][["actor","layer"]].drop_duplicates().shape[0] )
        stats_df = df(stats)
        
        # For comm_df, we need to concatenate all method dataframes
        comm_dfs = []
        for method_name, method_df in comm.items():
            method_df_copy = method_df.copy()
            method_df_copy['method'] = method_name
            comm_dfs.append(method_df_copy)
        comm_df = pd.concat(comm_dfs, ignore_index=True) if comm_dfs else pd.DataFrame()
        print("Community detection results:")
        print(stats_df)
           # Save communities and stats
        print("Saving results...")
        comm_df.to_csv(str(Path(__file__).resolve().parents[1] / "data" / "processed" / f"{experiment_name}_communities.csv"), index=False)
        stats_df.to_csv(str(Path(__file__).resolve().parents[1] / "data" / "processed" / f"{experiment_name}_stats.csv"), index=False)
    else:
        # load from memory
        comm_df = pd.read_csv(str(Path(__file__).resolve().parents[1] / "data" / "processed" / f"{experiment_name}_communities.csv"))
        stats_df = pd.read_csv(str(Path(__file__).resolve().parents[1] / "data" / "processed" / f"{experiment_name}_stats.csv"))
        
        print("Loaded community detection results from disk.")
            
    if args.do_plot:
        # Visualize the nodes and the results
        node_names = adata.obs_names.tolist()
        print("Visualizing the nodes and the results...")
        
        # For each actor, find the most frequently assigned community ID (highest voted)
        actor_cid_counts = comm_df.groupby(['actor', 'cid']).size().reset_index(name='count')
        
        # Get the highest voted cid for each actor
        highest_voted = actor_cid_counts.loc[actor_cid_counts.groupby('actor')['count'].idxmax()]
        
        # Create a mapping from actor to most frequent cid
        actor_to_cid = dict(zip(highest_voted['actor'], highest_voted['cid']))
        
        # Create df_to_plot with obs names and their corresponding highest voted cid
        df_to_plot = pd.DataFrame({
            "obs_name": node_names,
            "cid": [actor_to_cid.get(actor, -1) for actor in node_names],  # -1 for actors not found
            "centroid_x": adata.obs["centroid_x"].values,
            "centroid_y": adata.obs["centroid_y"].values
        })
        
        # Filter out actors that weren't found in community detection results
        df_to_plot = df_to_plot[df_to_plot["cid"] != -1]
        
        print(f"Created plotting dataframe with {len(df_to_plot)} observations")
        print(df_to_plot["cid"].value_counts())

        # Plot the results
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df_to_plot, x="centroid_x", y="centroid_y", hue="cid", palette="Set1", s=50, alpha=0.7)
        plt.title(f"Community Detection Results for {experiment_name}")
        plt.xlabel("Centroid X")
        plt.ylabel("Centroid Y")
        plt.show()

 
if __name__ == "__main__":
    main()