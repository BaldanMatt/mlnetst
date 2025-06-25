from pathlib import Path
import os
import pandas as pd

def load_resource(name: str, force: bool = False) -> pd.DataFrame:
    if name == "mouseconsensus":
        import liana as li
        print("Thanks for choosing the mouseconsensus from liana")
        lr_consensus = li.resource.select_resource("mouseconsensus")
        # rename ligand and receptor to source and target
        lr_consensus.rename(columns={"ligand": "source", "receptor": "target"}, inplace=True)
        return lr_consensus
    elif name == "nichenet":
        print("Thanks for choosing the nichenet resource")
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
        nichenet_net = pd.concat([lr_sig, gr_df], ignore_index=True)
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
        net = dc.op.collectri(
            organism="mouse",
            remove_complexes=False,
            )
        return net
    else:
        raise NotImplementedError("Requested resource is not available yet")
