from typing import List, Set, Dict, Tuple
from pathlib import Path
import os
import pandas as pd
def create_matches_intersections(dict_of_sets: Dict[str,Set]):
    """
    Create intersections of matches from a list of sets.

    Args:
        dict_of_sets (Dict[Set]): A dict containing sets of matches.

    Returns:
        Dict[Tuple, Set]: A dictionary where keys are tuples of keys of the sets,
                           and values are the intersection of the corresponding sets.
    """
    matches_intersections = {}
    sets_list = list(dict_of_sets.values())
    
    for i in range(len(sets_list)):
        for j in range(i + 1, len(sets_list)):
            intersection = sets_list[i].intersection(sets_list[j])
            if intersection:
                matches_intersections[(tuple(dict_of_sets.keys())[i], tuple(dict_of_sets.keys())[j])] = intersection    
    return matches_intersections

def map_human_to_mouse(
        dataframe,
        columns_to_translate,
) -> pd.DataFrame:
    """
    Translate human gene identifiers to mouse gene symbols using Homologene orthology.
    Args:
        dataframe (pd.DataFrame): DataFrame containing human gene identifiers.
        columns_to_translate (List[str]): List of column names in the DataFrame to translate.
    Returns:
        pd.DataFrame: DataFrame with human gene identifiers translated to mouse gene symbols.
    """

    from pypath.utils.orthology import HomologeneOrthology
    orthology_instance = HomologeneOrthology(
        target="mouse",
        source="human",
    )

    translated_df = orthology_instance.translate_df(
                dataframe,
                cols=columns_to_translate,
            )
    map_uniprot_to_gene_symbol = pd.read_csv(Path(__file__).parents[2] / "data" / "raw" / "idmap.csv")
    map_uniprot_to_gene_symbol.index = map_uniprot_to_gene_symbol["query"]
    map_uniprot_to_gene_symbol = map_uniprot_to_gene_symbol["symbol"].to_dict()
    for col in columns_to_translate:
        translated_df[col] = translated_df[col].apply(lambda x: map_uniprot_to_gene_symbol.get(x, x))

    return translated_df


