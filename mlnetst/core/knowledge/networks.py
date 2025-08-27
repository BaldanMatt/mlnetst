from pathlib import Path
import os
import pandas as pd

def load_resource(name: str, force: bool = False, extra_params: dict = {}) -> pd.DataFrame:
    """
    Load a specific resource based on the provided name and standardize return metadata.

    Args:
        name (str): The name of the resource to load.
        force (bool, optional): Whether to force reload the resource. Defaults to False.

    Raises:
        NotImplementedError: If the resource is not implemented.

    Returns:
        pd.DataFrame: The loaded resource as a DataFrame.
            columns: ["source", "target", "weight", "provenance", "interaction_type"]
    """
    
    
    
    if name == "mouseconsensus":
        import liana as li
        print("Thanks for choosing the mouseconsensus from liana")
        lr_consensus = li.resource.select_resource("mouseconsensus")
        # rename ligand and receptor to source and target
        lr_consensus.rename(columns={"ligand": "source", "receptor": "target"}, inplace=True)
        return lr_consensus
    elif name == "nichenet":
        print("Thanks for choosing the nichenet resource")
        which_provenance = extra_params.get("provenance", "all")
        if which_provenance not in ["all", "lr_sig", "gr"]:
            raise ValueError("Invalid provenance specified. Choose from 'all', 'lr_sig', or 'gr'.")
        
        gr_df = pd.read_csv(
            Path(__file__).parents[3] / "data" / "raw" / "gr.csv"
        )
        lr_sig = pd.read_csv(
            Path(__file__).parents[3] / "data" / "raw" / "lr_sig.csv"
        )
        # Rename ligand and receptor to source and target
        lr_sig.rename(columns={"from": "source", "to": "target"}, inplace=True)
        # Rename gene and target to source and target
        gr_df.rename(columns={"from": "source", "to": "target"}, inplace=True)
        # Merge the two dataframes with a column that contains from which resource the interaction comes from
        lr_sig["provenance"] = "nichenet_lr_sig"
        gr_df["provenance"] = "nichenet_gr"
        # Concatenate the two dataframes
        if which_provenance == "all":
            print("Combining ligand-receptor and gene regulatory interactions")
            nichenet_net = pd.concat([lr_sig, gr_df], ignore_index=True)
        elif which_provenance == "lr_sig":
            print("Using only ligand-receptor interactions")
            nichenet_net = lr_sig
        elif which_provenance == "gr":
            print("Using only gene regulatory interactions")
            nichenet_net = gr_df
        return nichenet_net
    elif name == "geneprograms":
        print("Thanks for choosing the niche compass gene programs")
        import nichecompass as nc
        if os.path.exists(Path(__file__).parents[3] / "data" / "raw" / "nichecompass_net_mouse.csv") and not force:
            nichecompass_net = pd.read_csv(
                Path(__file__).parents[3] / "data" / "raw" / "nichecompass_net_mouse.csv"
            )
        else:
            # nichecompass GPs (source: ligand genes; target: receptor genes, target genes)
            lrt_interactions = nc.utils.extract_gp_dict_from_nichenet_lrt_interactions(
                        species = "mouse",
                        version="v2",
                        keep_target_genes_ratio=1.,
                        load_from_disk = False,
                        save_to_disk = True,
                        lr_network_file_path=Path(__file__).parents[3] / "data" / "raw" / "nichenet_lr_network_mouse_v2.csv",
                        ligand_target_matrix_file_path=Path(__file__).parents[3] / "data" / "raw" / "nichenet_ligand_target_mouse_matrix_v2.csv",
                        plot_gp_gene_count_distributions = False,
            )
            lrt_interactions_df = pd.DataFrame.from_dict(lrt_interactions, orient="index")

            # omnipath GPs (source: ligand_genes; target: receptor_genes)
            lr_interactions = nc.utils.extract_gp_dict_from_omnipath_lr_interactions(
                        species="mouse",
                        gene_orthologs_mapping_file_path=Path(__file__).parents[3] / "data" / "raw" / "human_mouse_gene_orthologs.csv",
                        load_from_disk = False,
                        save_to_disk = True,
                        lr_network_file_path=Path(__file__).parents[3] / "data" / "raw" / "omnipath_lr_network.csv",
                        plot_gp_gene_count_distributions=False,
            )
            lr_interactions_df = pd.DataFrame.from_dict(lr_interactions, orient="index")
            # mebocost GPs (source: enzyme genes; target: sensor genes)
            es_interactions = nc.utils.extract_gp_dict_from_mebocost_es_interactions(
                        species="mouse",
                        plot_gp_gene_count_distributions=False,
                        dir_path=str(Path(__file__).parents[3] / "data" / "raw"),
            )
            es_interactions_df = pd.DataFrame.from_dict(es_interactions, orient="index")
            # collectri GPs (source: transcription factor genes; target: target genes)
            tf_interactions = nc.utils.extract_gp_dict_from_collectri_tf_network(
                        species="mouse",
                        tf_network_file_path=Path(__file__).parents[3] / "data" / "raw" / "collectri_tf_network_mouse.csv",
                        plot_gp_gene_count_distributions=False,
            )
            tf_interactions_df = pd.DataFrame.from_dict(tf_interactions, orient="index")
            # Add provenance
            lrt_interactions_df["provenance"] = "nichecompass_lrt"
            lr_interactions_df["provenance"] = "nichecompass_lr"
            es_interactions_df["provenance"] = "nichecompass_es"
            tf_interactions_df["provenance"] = "nichecompass_tf"
            # Combine the dictionaries 
            combined_gp_dict = nc.utils.filter_and_combine_gp_dict_gps(
                    [lrt_interactions, lr_interactions, es_interactions, tf_interactions],
                    verbose=True,
            )
            print(f"Number of gene programs: {len(combined_gp_dict)}")
            
            # Concatenate the dataframes
            nichecompass_net = pd.concat(
                [lrt_interactions_df, lr_interactions_df, es_interactions_df, tf_interactions_df],
                ignore_index=True
            )
            # Rename columns to source and target
            nichecompass_net.rename(columns={"sources": "source", "targets": "target"}, inplace=True)
            # Save the dataframe to a csv file
            nichecompass_net.to_csv(
                Path(__file__).parents[3] / "data" / "raw" / "nichecompass_net_mouse.csv",
                index=False
            )
        return nichecompass_net

    elif name == "omnipath":
        print("Thanks for choosing the omnipath resource")
        try:
            translated_omni_net = pd.read_csv(
                Path(__file__).parents[3] / "data" / "raw" / "omni_net_mouse.csv"
            )
        except FileNotFoundError:
            from pypath import omnipath
            from pypath import core
            omni_net = pd.read_csv(Path(__file__).parents[3]/ "data" / "raw" / "omni_net_human.csv")
            from mlnetst.utils.knowledge_utils import map_human_to_mouse
            translated_omni_net = map_human_to_mouse(
                omni_net,
                columns_to_translate=["id_a", "id_b"]
            )
            #Rename id_a and id_b to source and target
            translated_omni_net.rename(columns={"id_a": "source", "id_b": "target"}, inplace=True)
            translated_omni_net.to_csv(
                Path(__file__).parents[3] / "data" / "raw" / "omni_net_mouse.csv",
                index=False
            )
        return translated_omni_net 
    elif name == "collectri":
        print("thanks for choosing the collectri resource")
        import decoupler as dc
        net = dc.get_collectri(
            organism="mouse",
            split_complexes=False,
        )

    elif name == "scseqcomm":
        print("thanks for choosing the scseqcomm resource")
        import decoupler as dc
        tf_tg_collectri = dc.get_collectri(
            organism="mouse",
            split_complexes=False,
            
        )

        tf_tg_trrust = pd.read_csv(
            Path(__file__).parents[3] / "data" / "raw" / "TF_TG_TRRUSTv2_RegNetwork_High_mouse.csv"
        )
        rec_tf_kegg = pd.read_csv(
            Path(__file__).parents[3] / "data" / "raw" / "TF_PPR_KEGG_mouse.csv"
        )
        rec_tf_reactome = pd.read_csv(
            Path(__file__).parents[3] / "data" / "raw" / "TF_PPR_REACTOME_mouse.csv"
        )
        
        # Rename columns to source and target
        tf_tg_collectri.rename(columns={"source": "source", "target": "target"}, inplace=True)
        tf_tg_trrust.rename(columns={"tf": "source", "tg": "target"}, inplace=True)
        rec_tf_kegg.rename(columns={"receptor": "source", "tf": "target", "tf_PPR": "weight"}, inplace=True)
        rec_tf_reactome.rename(columns={"receptor": "source", "tf": "target", "tf_PPR": "weight"}, inplace=True)
        tf_tg_collectri.to_csv(
            Path(__file__).parents[3] / "data" / "raw" / "TF_TG_collectri_mouse.csv",
            index=False
        )
        net = pd.concat(
            [tf_tg_collectri, tf_tg_trrust, rec_tf_kegg, rec_tf_reactome],
            ignore_index=True
        )
        return net
        
    else:
        raise NotImplementedError("Requested resource is not available yet")

if __name__ == "__main__":
    resource = "scseqcomm"
    df = load_resource(resource, force=True)
    print(f"Loaded {resource} resource with {df.shape[0]} interactions.")
    print(df.head())
    print(df.columns)
    print(df.info())
    print(df["source"].nunique(), "unique sources")
    print(df["target"].nunique(), "unique targets")
