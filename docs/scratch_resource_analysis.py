import marimo

__generated_with = "0.13.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import anndata as ad
    import sys
    from pathlib import Path
    import scanpy as sc
    import networkx as nx
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(Path(__file__).parents[1]))
    print(sys.path)
    from mlnetst.core.knowledge.networks import load_resource
    return Path, ad, load_resource, nx, pd, plt, sc, sns


@app.cell
def _(Path, ad, load_resource):
    data_raw = ad.read_h5ad(Path(__file__).parents[1] / "data" / "processed" / "mouse1_slice153_x_hat_s.h5ad")
    lr_interactions = load_resource("mouseconsensus")
    nichenet_v2 = load_resource("nichenet")
    return data_raw, lr_interactions, nichenet_v2


@app.cell
def _(data_raw, lr_interactions, nichenet_v2, sc):
    data = data_raw.copy()
    min_gene_expression = 0
    min_cells_per_gene = 100
    min_genes_per_cell = 3
    data.X[data.X < min_gene_expression] = 0
    sc.pp.filter_genes(data, min_cells=min_cells_per_gene)
    sc.pp.filter_cells(data, min_genes=min_genes_per_cell)
    valid_genes = data.var_names.tolist()
    ligand_ids = lr_interactions["source"].str.lower().unique().tolist()
    source_ids = nichenet_v2["source"].str.lower().unique().tolist()
    receptor_ids = lr_interactions["target"].str.lower().unique().tolist()
    target_ids = nichenet_v2["target"].str.lower().unique().tolist()
    source_ids_only_grn = nichenet_v2.query("provenance == 'nichenet_gr'")["source"].str.lower().unique().tolist()
    target_ids_only_grn = nichenet_v2.query("provenance == 'nichenet_gr'")["target"].str.lower().unique().tolist()
    return (
        ligand_ids,
        receptor_ids,
        source_ids,
        source_ids_only_grn,
        target_ids,
        target_ids_only_grn,
        valid_genes,
    )


@app.cell
def _(
    ligand_ids,
    receptor_ids,
    source_ids,
    source_ids_only_grn,
    target_ids,
    target_ids_only_grn,
    valid_genes,
):
    num_valid_genes_that_are_ligands = len(set(valid_genes).intersection(set(ligand_ids)))
    num_valid_genes_that_are_receptors = len(set(valid_genes).intersection(set(receptor_ids)))
    num_valid_genes_that_are_both = len(set(valid_genes).intersection(set(ligand_ids)).intersection(set(receptor_ids)))
    num_valid_genes_that_are_source = len(set(valid_genes).intersection(set(source_ids)))
    num_valid_genes_that_are_targets = len(set(valid_genes).intersection(set(target_ids)))
    num_valid_genes_that_are_both_nich = len(set(valid_genes).intersection(set(source_ids)).intersection(set(target_ids)))
    num_valid_genes_that_are_source_grn_only = len(set(valid_genes).intersection(set(source_ids_only_grn)))
    num_valid_genes_that_are_targets_grn_only = len(set(valid_genes).intersection(set(target_ids_only_grn)))
    num_valid_genes_that_are_both_nich_grn_only = len(set(valid_genes).intersection(set(source_ids_only_grn)).intersection(set(target_ids_only_grn)))
    num_receptors_that_are_source = len(set(receptor_ids).intersection(set(source_ids)))
    num_receptors_that_are_source_grn_only = len(set(receptor_ids).intersection(set(source_ids_only_grn)))
    num_ligands_that_are_target = len(set(ligand_ids).intersection(set(target_ids)))
    num_ligands_that_are_target_grn_only = len(set(ligand_ids).intersection(set(target_ids_only_grn)))
    num_receptor_that_are_both = len(set(receptor_ids).intersection(set(source_ids)).intersection(source_ids_only_grn))
    num_targets_that_are_both = len(set(ligand_ids).intersection(set(target_ids)).intersection(target_ids_only_grn))



    print(f"Num genes that are ligands: {num_valid_genes_that_are_ligands} out of valid genes [ligands] {len(valid_genes)}[{len(set(ligand_ids))}]\n\t \
    Num genes that are receptors: {num_valid_genes_that_are_receptors} out of valid genes [receptors] {len(valid_genes)}[{len(set(receptor_ids))}]\n\t \
    Num genes that are both: {num_valid_genes_that_are_both} out of valid genes [both] {len(valid_genes)}[{len(set(receptor_ids).intersection(set(ligand_ids)))}]" )

    print(f"Num genes that are source: {num_valid_genes_that_are_source} out of valid genes [source] {len(valid_genes)}[{len(set(source_ids))}]\n\t \
    Num genes that are target: {num_valid_genes_that_are_targets} out of valid genes [target] {len(valid_genes)}[{len(set(target_ids))}]\n\t \
    Num genes that are both: {num_valid_genes_that_are_both_nich} out of valid genes [both] {len(valid_genes)}[{len(set(target_ids).intersection(set(source_ids)))}]" )

    print(f"Num genes that are source in grn: {num_valid_genes_that_are_source_grn_only} out of valid genes [source] {len(valid_genes)}[{len(set(source_ids_only_grn))}]\n\t \
    Num genes that are target in grn: {num_valid_genes_that_are_targets_grn_only} out of valid genes [target] {len(valid_genes)}[{len(set(target_ids_only_grn))}]\n\t \
    Num genes that are both: {num_valid_genes_that_are_both_nich_grn_only} out of valid genes [both] {len(valid_genes)}[{len(set(target_ids_only_grn).intersection(set(source_ids_only_grn)))}]" )

    print(f"Num receptors that are source: {num_receptors_that_are_source} out of receptors [source] {len(set(receptor_ids))}[{len(set(source_ids))}]\n\t \
    Num receptors that are source in grn: {num_receptors_that_are_source_grn_only} out of valid genes [target] {len(set(receptor_ids))}[{len(set(source_ids_only_grn))}]\n\t \
    Num receptors that are both: {num_receptor_that_are_both} out of valid genes [both] {len(set(receptor_ids))}[{len(set(source_ids).intersection(set(source_ids_only_grn)))}]" )

    print(f"Num ligands that are targets: {num_ligands_that_are_target} out of ligands [targets] {len(set(ligand_ids))}[{len(set(target_ids))}]\n\t \
    Num ligands that are target in grn: {num_ligands_that_are_target_grn_only} out of ligands [targets] {len(set(ligand_ids))}[{len(set(target_ids_only_grn))}]\n\t \
    Num ligands that are both: {num_targets_that_are_both} out of ligands [both] {len(set(ligand_ids))}[{len(set(target_ids).intersection(set(target_ids_only_grn)))}]" )
    return


@app.cell
def _(nichenet_v2, nx):
    net_df = nichenet_v2.copy()
    net_df["source"] = net_df["source"].str.lower()
    net_df["target"] = net_df["target"].str.lower()
    net_df = net_df.query("provenance == 'nichenet_gr'")
    net = nx.from_pandas_edgelist(
                net_df,
                source='source',
                target='target',
                create_using=nx.DiGraph
            )
    return net, net_df


@app.cell
def _(net):
    import numpy as np
    print(net.degree)
    idx_max = np.argmax([d for n,d in net.degree])
    print(idx_max)
    print(list(net.degree)[idx_max])
    return (np,)


@app.cell
def _(net, net_df, np, plt, sns):
    print(net)
    net_df["sourcetarget"] = net_df["source"]+net_df["target"]
    net_df["targetsource"] = net_df["target"]+net_df["source"]
    net_df["is_bidir"] = net_df["sourcetarget"] == net_df["targetsource"]
    print(net_df["is_bidir"].value_counts())
    from sklearn.preprocessing import MinMaxScaler

    # Better approach for min-max normalization and visualization

    # Method 1: Using sklearn's MinMaxScaler (recommended)
    def normalize_with_sklearn(df, column):
        """Normalize using sklearn's MinMaxScaler"""
        scaler = MinMaxScaler()
        return scaler.fit_transform(df[[column]])
    scaler = MinMaxScaler()
    net_df["weight_minmax"] = normalize_with_sklearn(net_df, 'weight')
    print("Done with minmaxing...")
    fig, axs = plt.subplots(nrows=4)
    sns.violinplot(net_df, x="weight",ax=axs[0])
    sns.violinplot(net_df, x="weight_minmax",ax=axs[1])
    axs[2].hist(dict(net.degree()).values())
    axs[3].hist(scaler.fit_transform(np.array(list(net.degree))[:, 1].reshape(-1, 1)).flatten(), bins=50)
    plt.show()


    return


@app.cell
def _(nx, plt):
    # Create a directed graph
    G = nx.DiGraph()

    # Add 5 nodes
    nodes = [1, 2, 3, 4, 5]
    G.add_nodes_from(nodes)

    # Add edges to connect 4 nodes, leaving node 5 isolated
    edges = [
        (1, 2),
        (2, 3),
        (3, 4),
        (1, 4)
    ]
    G.add_edges_from(edges)

    # Print graph information
    print("Graph Information:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Nodes: {list(G.nodes())}")
    print(f"Edges: {list(G.edges())}")
    print(f"Isolated nodes: {list(nx.isolates(G))}")
    # Visualize the graph
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=1000, font_size=16, font_weight='bold',
            arrows=True, arrowsize=20, edge_color='gray')
    plt.title("DiGraph with 5 Nodes (Node 5 is Isolated)")
    plt.axis('off')
    plt.show()
    return


@app.cell
def _(net, nx):
    a = set(net.nodes).difference(set(nx.descendants(net, "acvr2a")))
    b = set(net.nodes).difference(set(nx.descendants(net, "fgfr1")))
    print(a)
    print(b)
    print(len(set(nx.descendants(net, "ifx"))))
    return


@app.cell
def _(Path, pd):
    tf_ppr_kegg_mouse = pd.read_csv(str(Path(__file__).parents[1] / "data" / "raw" / "TF_PPR_KEGG_mouse.csv"))
    print(tf_ppr_kegg_mouse.head(3))
    tf_ppr_reactome_mouse = pd.read_csv(str(Path(__file__).parents[1] / "data" / "raw" / "TF_PPR_REACTOME_mouse.csv"))
    print(tf_ppr_reactome_mouse.head(3))
    tf_tg_trrustv2 = pd.read_csv(str(Path(__file__).parents[1] / "data" / "raw" / "TF_TG_TRRUSTv2_RegNetwork_High_mouse.csv"))
    print(tf_tg_trrustv2.head(3))
    import decoupler as dc
    collectri = dc.get_collectri(organism="mouse")
    print(collectri)
    return collectri, tf_ppr_kegg_mouse, tf_ppr_reactome_mouse, tf_tg_trrustv2


@app.cell
def _(
    collectri,
    lr_interactions,
    tf_ppr_kegg_mouse,
    tf_ppr_reactome_mouse,
    tf_tg_trrustv2,
):
    num_receptor_in_kegg_and_reactome = len(set(tf_ppr_kegg_mouse["receptor"]).intersection(set(tf_ppr_reactome_mouse["receptor"])))
    print(f"Num of receptors in kegg and reactome: {num_receptor_in_kegg_and_reactome} out of {len(set(tf_ppr_kegg_mouse["receptor"]))} and {len(set(tf_ppr_reactome_mouse["receptor"]))}")
    num_tf_in_kegg_and_reactome = len(set(tf_ppr_kegg_mouse["tf"]).intersection(set(tf_ppr_reactome_mouse["tf"])))
    print(f"Num of tf in kegg and reactome: {num_tf_in_kegg_and_reactome} out of {len(set(tf_ppr_kegg_mouse["tf"]))} and {len(set(tf_ppr_reactome_mouse["tf"]))}")
    num_tg_in_trrust_and_collectri = len(set(tf_tg_trrustv2["tg"]).intersection(set(collectri["target"])))
    num_tf_in_trrust_and_collectri = len(set(tf_tg_trrustv2["tf"]).intersection(set(collectri["source"])))
    print(f"Num of tf in trrust and collectr: {num_tf_in_trrust_and_collectri} out of {len(set(tf_tg_trrustv2["tf"]))} and {len(set(collectri["source"]))}")
    print(f"Num of tg in trrust and collectr: {num_tg_in_trrust_and_collectri} out of {len(set(tf_tg_trrustv2["tg"]))} and {len(set(collectri["target"]))}")
    num_tf_ppr_in_resources = len(set(tf_ppr_kegg_mouse["tf"]).intersection(set(tf_ppr_reactome_mouse["tf"])).intersection(set(tf_tg_trrustv2["tf"])).intersection(set(collectri["source"])))
    print(f"Num of tf in all res: {num_tf_ppr_in_resources} out of {len(set(tf_ppr_kegg_mouse["tf"]))}, {len(set(tf_ppr_reactome_mouse["tf"]))}, {len(set(tf_tg_trrustv2["tf"]))} and {len(set(collectri["source"]))}")
    num_ligands_in_tgs = len(set(lr_interactions["source"].unique().tolist()).intersection(set(tf_tg_trrustv2["tg"])).intersection(set(collectri["target"])))
    print(f"Num of ligands in all res: {num_ligands_in_tgs} out of {len(set(lr_interactions["source"].unique().tolist()))}, {len(set(tf_tg_trrustv2["tg"]))} and {len(set(collectri["target"]))}")
    return


if __name__ == "__main__":
    app.run()
