import sys
from pathlib import Path
import os
import anndata as ad
sys.path.append(str(Path(__file__).parents[1]))

from mlnetst.core.knowledge.networks import load_resource
from mlnetst.utils.knowledge_utils import create_matches_intersections

import argparse
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="Name of the resource you want to load")
    args = parser.parse_args()
    return args

def main(name):
    # Preparing all resources that i want to use to compare intersections
    adata = ad.read_h5ad(
            Path(__file__).parents[1] / "data" / "processed" / "mouse1_slice153_x_hat_s.h5ad"
    )
    # Change all var_names to have the first letter capitalized
    adata.var_names = [name.capitalize() for name in adata.var_names]
    net_list_mousecons = load_resource("mouseconsensus")
    net_list_omni = load_resource("omnipath")
    net_list_nichenet = load_resource("nichenet")
    net_list_nichecompass = load_resource("geneprograms", force=True)

    # Start from here we analyse resources
    print("Creating matches intersections")
    matches = create_matches_intersections(
        {
            f"ligand_mouse_cons": set(net_list_mousecons["source"]),
            f"receptor_mouse_cons": set(net_list_mousecons["target"]),
            f"source_omnipath": set(net_list_omni["source"]),
            f"target_omnipath": set(net_list_omni["target"]),
            f"genes_adata": set(adata.var_names),
            f"source_nichenet": set(net_list_nichenet["source"]),
            f"target_nichenet": set(net_list_nichenet["target"]),
            f"source_nichecompass": set(net_list_nichecompass["source"]),
            f"target_nichecompass": set(net_list_nichecompass["target"]),
        }
    )
    print("Matches intersections created:")
    for key, value in matches.items():
        print(f"{key}: {len(value)} matches")

    import matplotlib.pyplot as plt
    from matplotlib_venn import venn3
    # Create a Venn diagram
    testable_sets = {
        ("Ligand mouse consensus", "Receptor Mouse Consensus", "X hat s genes"):(net_list_mousecons["source"], net_list_mousecons["target"], adata.var_names),
        ("Source omnipath","Target omnipath", "X hat s genes"):(net_list_omni["source"], net_list_omni["target"], adata.var_names),
        ("Ligand mouse consensus", "Source omnipath", "X hat s genes"):(net_list_mousecons["source"], net_list_omni["source"], adata.var_names),
        ("Receptor Mouse Consensus", "Target omnipath", "X hat s genes"):(net_list_mousecons["target"], net_list_omni["target"], adata.var_names),
        ("Ligand mouse consensus", "Receptor Mouse Consensus", "Source omnipath"):(net_list_mousecons["source"], net_list_mousecons["target"], net_list_omni["source"]),
        ("Ligand mouse consensus", "Receptor Mouse Consensus", "Target omnipath"):(net_list_mousecons["source"], net_list_mousecons["target"], net_list_omni["target"]),
        ("Source omnipath", "Target omnipath", "X hat s genes"):(net_list_omni["source"], net_list_omni["target"], adata.var_names),
        ("Source Nichenet", "Target Nichenet", "X hat s genes"):(net_list_nichenet["source"], net_list_nichenet["target"], adata.var_names),
        ("Source Nichenet", "Target Nichenet", "Source omnipath"):(net_list_nichenet["source"], net_list_nichenet["target"], net_list_omni["source"]),
        ("Source Nichenet", "Target Nichenet", "Target omnipath"):(net_list_nichenet["source"], net_list_nichenet["target"], net_list_omni["target"]),
    }
    for iplot, (setnames, triplet) in enumerate(testable_sets.items()):
        print(f"Plotting {iplot} venn Diagram for sets: {setnames}")
        plt.figure(figsize=(10, 7))
        a, b, c = triplet
        name_a, name_b, name_c = setnames
        venn3(subsets=(
                set(a),
                set(b),        
                set(c),
        ),
            set_labels=(name_a, name_b, name_c),)
        plt.title("Venn Diagram of Matches Intersections")
        plt.savefig(Path(__file__).parents[1] / "data" / "media" / f"{iplot}_matches_intersections.png")

    # Are there emergent patterns

if __name__ == "__main__":
    args = parse_arguments()
    main(args.name)
