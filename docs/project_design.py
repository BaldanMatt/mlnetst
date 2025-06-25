import marimo

__generated_with = "0.13.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sys
    from pathlib import Path
    import os
    import nichecompass as nc
    import pandas as pd
    import anndata as ad
    import numpy as np
    import torch
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
    import random

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    pd.core.common.random_state(None)
    random.seed(RANDOM_SEED)
    return Path, ad, nc, np, nx, pd, plt, sys, torch


@app.cell
def _(Path, ad):
    x_hat_s = ad.read_h5ad(Path(__file__).parents[1] / "data" / "processed" / "mouse1_slice153_x_hat_s.h5ad")
    print(x_hat_s)
    return (x_hat_s,)


@app.cell
def _(x_hat_s):
    source, target = ("Astro", "L2/3 IT")
    subdata = x_hat_s[x_hat_s.obs["subclass"].isin([source,target]),:]
    print(subdata)
    return (subdata,)


@app.cell
def _(lr_interactions_df, subdata):
    # Sample 1 LR interactions from mouse consensus
    notFound = True
    L = 5
    while notFound:
        sample_lr = lr_interactions_df.sample(n=L)
        if list(sample_lr["sources"][0])[0].lower() in subdata.var_names and list(sample_lr["targets"][0])[0].lower() in subdata.var_names:
            notFound = False
    print(sample_lr)
    return L, sample_lr


@app.cell
def _(sample_lr):
    list(sample_lr["sources"].explode())[0]
    return


@app.cell
def _(L, np, nx, plt, sample_lr, subdata, torch):
    id_lr = 0
    TOLL_DIST = 1e-5
    TH_INTERCELLULAR = 0.975
    N = subdata.shape[0]
    # Add spatial coordinates
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

    # Init sparse coo tensor
    mlnet = torch.sparse_coo_tensor(size=(N,N,L))
    print(mlnet)

    # Select intralayer mask L
    ligand_mask = torch.from_numpy(
        subdata[:, list(sample_lr["sources"].explode())[id_lr].lower()].X.astype(np.float32)
    )
    # Select intralayer mask R
    receptor_mask = torch.from_numpy(
        subdata[:, list(sample_lr["targets"].explode())[id_lr].lower()].X.astype(np.float32)
    )


    # Compute intralayer score
    single_lr = torch.div(ligand_mask @ receptor_mask.T, dist_matrix + TOLL_DIST)
    single_lr = single_lr.fill_diagonal_(0)

    # Threshold naively to have a sparse network
    mask = single_lr > np.quantile(single_lr.numpy().flatten(), TH_INTERCELLULAR)
    masked_single_lr = single_lr * mask
    print(f"State of the single intralayer: {single_lr}")

    # Cast to convert to network object and draw
    cells_inter_scores = masked_single_lr.to_sparse()

    mlnet = torch.dstack((mlnet, cells_inter_scores))
    print(mlnet)

    adj = cells_inter_scores.to_dense().numpy()
    g = nx.from_numpy_array(adj)
    print(f"State of the intralayer: {g}")
    pos = [(x,y) for x,y in zip(spatial_position_tensor[:,0].numpy(), spatial_position_tensor[:,1].numpy())]
    nx.draw(g, pos, node_size=10, node_color="blue", alpha=0.5)
    plt.title(f"Ligand {sample_lr["sources"].explode()[id_lr].lower()} - Receptor {sample_lr["targets"].explode()[id_lr].lower()}")
    plt.show()
    return (g,)


@app.cell
def _(g, plt):
    print(g)
    degrees = [deg for _, deg in g.degree()]
    print(degrees)
    plt.hist(degrees, bins=20)
    plt.show()
    return


@app.cell
def _(Path, sys):
    sys.path.append(str(Path(__file__).parents[1]))
    return


@app.cell
def _():
    from mlnetst.core.knowledge.networks import load_resource
    return


@app.cell
def _(Path, nc, pd):
    # nichecompass GPs (source: ligand genes; target: receptor genes, target genes)
    lrt_interactions = nc.utils.extract_gp_dict_from_nichenet_lrt_interactions(
                            species = "mouse",
                            version="v2",
                            keep_target_genes_ratio=1.,
                            load_from_disk = True,
                            save_to_disk = False,
                            lr_network_file_path=Path(__file__).parents[1] / "data" / "raw" / "nichenet_lr_network_mouse_v2.csv",
                            ligand_target_matrix_file_path=Path(__file__).parents[1] / "data" / "raw" / "nichenet_ligand_target_mouse_matrix_v2.csv",
                            plot_gp_gene_count_distributions = False,
                )
    lrt_interactions_df = pd.DataFrame.from_dict(lrt_interactions, orient="index")
    return lrt_interactions, lrt_interactions_df


@app.cell
def _(lrt_interactions_df):
    lrt_interactions_df.sources.value_counts()
    return


@app.cell
def _(lrt_interactions_df):
    lrt_interactions_df
    return


@app.cell
def _(Path, nc, pd):
    # omnipath GPs (source: ligand_genes; target: receptor_genes)
    lr_interactions = nc.utils.extract_gp_dict_from_omnipath_lr_interactions(
                            species="mouse",
                            gene_orthologs_mapping_file_path=Path(__file__).parents[1] / "data" / "raw" / "human_mouse_gene_orthologs.csv",
                            load_from_disk = True,
                            save_to_disk = False,
                            lr_network_file_path=Path(__file__).parents[1] / "data" / "raw" / "omnipath_lr_network.csv",
                            plot_gp_gene_count_distributions=False,
                )
    lr_interactions_df = pd.DataFrame.from_dict(lr_interactions, orient="index")
    return lr_interactions, lr_interactions_df


@app.cell
def _(lr_interactions_df):
    lr_interactions_df.sources.value_counts()
    return


@app.cell
def _(Path, nc, pd):
    # mebocost GPs (source: enzyme genes; target: sensor genes)
    es_interactions = nc.utils.extract_gp_dict_from_mebocost_es_interactions(
                            species="mouse",
                            plot_gp_gene_count_distributions=False,
                            dir_path=str(Path(__file__).parents[1] / "data" / "raw"),
                )
    es_interactions_df = pd.DataFrame.from_dict(es_interactions, orient="index")
    return es_interactions, es_interactions_df


@app.cell
def _(
    Path,
    es_interactions_df,
    lr_interactions_df,
    lrt_interactions_df,
    nc,
    pd,
):
    # collectri GPs (source: transcription factor genes; target: target genes)
    tf_interactions = nc.utils.extract_gp_dict_from_collectri_tf_network(
                            species="mouse",
                            tf_network_file_path=Path(__file__).parents[1] / "data" / "raw" / "collectri_tf_network_mouse.csv",
                            plot_gp_gene_count_distributions=False,
                )
    tf_interactions_df = pd.DataFrame.from_dict(tf_interactions, orient="index")
                # Add provenance
    lrt_interactions_df["provenance"] = "nichecompass_lrt"
    lr_interactions_df["provenance"] = "nichecompass_lr"
    es_interactions_df["provenance"] = "nichecompass_es"
    tf_interactions_df["provenance"] = "nichecompass_tf"
    return (tf_interactions,)


@app.cell
def _(lrt_interactions):
    lrt_interactions["2300002M23Rik_ligand_receptor_target_gene_GP"]
    return


@app.function
def filter_and_combine_gp_dict_gps_v2(
        gp_dicts: list,
        overlap_thresh_target_genes: float=1.,
        verbose: bool=False) -> dict:
    """
    Combine gene program dictionaries and filter them based on gene overlaps.

    Parameters
    ----------
    gp_dicts:
        List of gene program dictionaries with keys being gene program names and
        values being dictionaries with keys ´sources´, ´targets´,
        ´sources_categories´, and ´targets_categories´, where ´targets´ contains
        a list of the names of genes in the gene program for the reconstruction
        of the gene expression of the node itself (receiving node) and ´sources´
        contains a list of the names of genes in the gene program for the
        reconstruction of the gene expression of the node's neighbors
        (transmitting nodes).
    overlap_thresh_target_genes:
        The minimum ratio of target genes that need to overlap between a GP
        without source genes and another GP for the GP to be dropped.
        Gene programs with different source genes are never combined or dropped.
    verbose:
        If `True`, print gene programs that are dropped and combined.

    Returns
    ----------
    new_gp_dict:
        Combined gene program dictionary with filtered gene programs.
    """
    # Combine gene program dictionaries
    combined_gp_dict = {}
    for i, gp_dict in enumerate(gp_dicts):
        combined_gp_dict.update(gp_dict)

    new_gp_dict = combined_gp_dict.copy()

    # Combine gene programs with overlapping genes
    all_combined = False
    while not all_combined:
        all_combined = True
        combined_gp_dict = new_gp_dict.copy()
        for i, (gp_i, gp_genes_dict_i) in enumerate(combined_gp_dict.items()):
            source_genes_i = [
                gene for gene in gp_genes_dict_i["sources"]]
            target_genes_i = [
                gene for gene in gp_genes_dict_i["targets"]]
            target_genes_categories_i = [
                target_gene_category for target_gene_category in
                gp_genes_dict_i["targets_categories"]]
            for j, (gp_j, gp_genes_dict_j) in enumerate(
                combined_gp_dict.items()):
                if j != i:
                    source_genes_j = [
                        gene for gene in gp_genes_dict_j["sources"]]
                    target_genes_j = [
                        gene for gene in gp_genes_dict_j["targets"]]
                    target_genes_categories_j = [
                        target_gene_category for target_gene_category in
                        gp_genes_dict_j["targets_categories"]]

                    if ((source_genes_i == source_genes_j) &
                        len(source_genes_i) > 0):
                        # if source genes are exactly the same, combine gene
                        # programs
                        all_combined = False
                        if verbose:
                            print(f"Combining {gp_i} and {gp_j}.")
                        source_genes = source_genes_i
                        target_genes = target_genes_i
                        target_genes_categories = target_genes_categories_i
                        for target_gene, target_gene_category in zip(
                            target_genes_j, target_genes_categories_j):
                            if target_gene not in target_genes:
                                target_genes.extend([target_gene])
                                target_genes_categories.extend(
                                    [target_gene_category])
                        new_gp_dict.pop(gp_i, None)
                        new_gp_dict.pop(gp_j, None)
                        if (gp_j.split("_")[0] + 
                            "_combined_GP") not in new_gp_dict.keys():
                            new_gp_name = gp_i.split("_")[0] + "_combined_GP"
                            new_gp_dict[new_gp_name] = {"sources": source_genes}
                            new_gp_dict[new_gp_name]["targets"] = target_genes
                            new_gp_dict[new_gp_name][
                                "sources_categories"] = gp_genes_dict_i[
                                    "sources_categories"]
                            new_gp_dict[new_gp_name][
                                "targets_categories"] = target_genes_categories

                    elif len(source_genes_i) == 0:
                        target_genes_overlap = list(
                            set(target_genes_i) & set(target_genes_j))
                        n_target_gene_overlap = len(target_genes_overlap)
                        n_target_genes = len(target_genes_i)
                        ratio_shared_target_genes = (n_target_gene_overlap /
                                                     n_target_genes)
                        if ratio_shared_target_genes >= overlap_thresh_target_genes:
                            # if source genes not existent and target genes
                            # overlap more than specified, drop gene program
                            if gp_j in new_gp_dict.keys():
                                if verbose:
                                    print(f"Dropping {gp_i}.")
                                new_gp_dict.pop(gp_i, None)
                    else:
                        # otherwise do not combine or drop gene programs
                        pass

    return new_gp_dict


@app.cell
def _(es_interactions, lr_interactions, lrt_interactions, tf_interactions):
    # Combine the dictionaries 
    combined_gp_dict = filter_and_combine_gp_dict_gps_v2(
        [lrt_interactions, lr_interactions, es_interactions, tf_interactions],
    )
    print(f"gene programs: {len(combined_gp_dict)}")

    return (combined_gp_dict,)


@app.cell
def _(Path, combined_gp_dict, pd):
    combined_gp_dict_df = pd.DataFrame.from_dict(combined_gp_dict, orient="index")
    combined_gp_dict_df.info()
    combined_gp_dict_df.to_csv(Path(__file__).parents[1] / "data" / "raw" / "nichecompass_combined_lrt_lr_es_tf.csv")

    return (combined_gp_dict_df,)


@app.cell
def _(combined_gp_dict_df):
    combined_gp_dict_df.sources_categories.value_counts()
    return


if __name__ == "__main__":
    app.run()
