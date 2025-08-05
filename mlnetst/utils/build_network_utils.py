import torch
import numpy as np
from typing import Tuple, List, Union, Dict, Any
import pandas as pd
import networkx as nx
import logging
from mlnetst.core.knowledge.networks import load_resource

def create_layer_gene_mapping(ligand_ids: List[str], receptor_ids: List[str], var_names: List[str]) -> Dict[int, Dict[str, Dict[str, List[int]]]]:
    """
    Create a unified mapping between layer indices and both ligand and receptor components.
    
    Args:
        ligand_ids: List of ligand IDs (can include complex genes separated by '_')
        receptor_ids: List of receptor IDs (can include complex genes separated by '_')
        var_names: List of variable names available in the dataset
        
    Returns:
        Dictionary mapping layer index to both ligand and receptor information:
        {
            layer_idx: {
                "ligand": {
                    "gene_id": original ligand id,
                    "component_indices": list of indices in var_names for each component
                },
                "receptor": {
                    "gene_id": original receptor id,
                    "component_indices": list of indices in var_names for each component
                }
            }
        }
    """
    layer_map = {}
    
    for layer_idx, (ligand_id, receptor_id) in enumerate(zip(ligand_ids, receptor_ids)):
        layer_info = {"ligand": {}, "receptor": {}}
        valid_layer = True
        
        # Process ligand
        ligand_components = ligand_id.split("_")
        ligand_indices = []
        for component in ligand_components:
            try:
                idx = var_names.tolist().index(component)
                ligand_indices.append(idx)
            except ValueError:
                valid_layer = False
                break
                
        # Process receptor
        receptor_components = receptor_id.split("_")
        receptor_indices = []
        for component in receptor_components:
            try:
                idx = var_names.tolist().index(component)
                receptor_indices.append(idx)
            except ValueError:
                valid_layer = False
                break
        
        # Only add to mapping if all components were found
        if valid_layer and ligand_indices and receptor_indices:
            layer_info["ligand"] = {
                "gene_id": ligand_id,
                "component_indices": ligand_indices
            }
            layer_info["receptor"] = {
                "gene_id": receptor_id,
                "component_indices": receptor_indices
            }
            layer_info["layer_name"] = f"{ligand_id}_{receptor_id}"
            layer_map[layer_idx] = layer_info
            
    return layer_map

def select_intra_layers(ligand_ids: List[str], var_names: List[str]) -> List[int]:
    """
    Select indices of layers based on ligand IDs.
    
    Args:
        ligand_ids: List of ligand IDs to select
        var_names: List of variable names (layer names)
        
    Returns:
        List of indices corresponding to the selected layers
    """
    return [var_names.tolist().index(ligand_id) for ligand_id in ligand_ids if ligand_id in var_names]

def extract_suitable_layers_from_net(net: nx.DiGraph, ligand_ids: List[str], receptor_ids: List[str]) -> Any:
    """
    This function wants to find the set of nodes that are within the set of Ligands that are reachable for each node of a network that is within Receptor set.
    """
    if not isinstance(net, nx.DiGraph):
        raise TypeError("Input must be a networkx DiGraph")
    if not isinstance(ligand_ids, list) or not isinstance(receptor_ids, list):
        raise TypeError("ligand_ids and receptor_ids must be lists")
    if not all(isinstance(x, str) for x in ligand_ids) or not all(isinstance(x, str) for x in receptor_ids):
        raise TypeError("All elements in ligand_ids and receptor_ids must be strings")
    
    # Find all nodes that are ligands that are reachable from any receptor node
    reachable_from_receptors = set()
    for r in net.nodes():
        if r in receptor_ids:
            # Get all nodes reachable from this receptor
            reachable_nodes = nx.descendants(net, r)
            # Add only those that are ligands
            reachable_from_receptors.update([n for n in reachable_nodes if n in ligand_ids])
    
    # Create pairs of receptor and reachable ligand nodes
    pairs = []
    for r in receptor_ids:
        if r in net.nodes():
            for l in reachable_from_receptors:
                if l in net.nodes():
                    pairs.append((r, l))

    return pairs

def check_complex_descendancy(net: nx.DiGraph, source_components: List[str], 
                            target_components: List[str]) -> bool:
    """
    Check if all source components have all target components in their descendancy.
    
    Args:
        net: NetworkX directed graph
        source_components: List of source gene components
        target_components: List of target gene components
        
    Returns:
        bool: True if all components satisfy the descendancy condition
    """
    # For each source component, get its descendants once
    source_descendants = {
        src: set(nx.descendants(net, src)) 
        for src in source_components
    }
    
    # Check if all source components can reach all target components
    return all(
        all(target in descendants 
            for target in target_components)
        for descendants in source_descendants.values()
    )

def select_inter_layers(suitable_pairs: pd.DataFrame,
                        layer_mapping: Dict[int, Dict[str, Dict[str, List[int]]]],
                        resource: str = "nichenet", 
                        inter_coupling: str = "combinatorial",
                        logger: logging.Logger | None = None) -> List[Tuple[int, int]]:
    """
    Select inter-layer pairs based on suitable pairs and layer mapping.
    
    Args:
        suitable_pairs: DataFrame of tuples (receptor, ligand) representing suitable pairs
        layer_mapping: Layer mapping dictionary as created by create_layer_gene_mapping
    Returns:
        List of tuples (src_layer, dst_layer) representing inter-layer pairs
    """
    inter_layer_pairs = []
    
    if inter_coupling == "combinatorial":    
        for receptor, ligand in suitable_pairs:
            # Find layers for receptor
            src_layers = [idx for idx, info in layer_mapping.items() if info["receptor"]["gene_id"] == receptor]
            # Find layers for ligand
            dst_layers = [idx for idx, info in layer_mapping.items() if info["ligand"]["gene_id"] == ligand]
            
            # Create pairs of (src_layer, dst_layer)
            for src in src_layers:
                for dst in dst_layers:
                    inter_layer_pairs.append((src, dst))
    
    elif inter_coupling == "rtl": # stands for receptor-to-tf-to-targetgene/ligand
        
        net_df = load_resource(resource)
        suitable_pairs[["source", "target"]] = suitable_pairs[["source", "target"]].apply(lambda x: x.str.lower())
        net_df[["source", "target"]] = net_df[["source", "target"]].apply(lambda x: x.str.lower())
        net = nx.from_pandas_edgelist(
            net_df,
            source='source',
            target='target',
            create_using=nx.DiGraph
        )
        logger.debug(f"Network loaded with {net.number_of_nodes()} nodes and {net.number_of_edges()} edges")
        completed_layers = 0
        num_layers = len(layer_mapping)
        for src_layer, src_info in layer_mapping.items():
            # Get source receptor components
            source_receptor = src_info["receptor"]["gene_id"]
            source_components = source_receptor.split("_")
            
            for dst_layer, dst_info in layer_mapping.items():
                if src_layer == dst_layer:
                    continue  # Skip self-loops
                    
                # Get target ligand components
                target_ligand = dst_info["ligand"]["gene_id"]
                target_components = target_ligand.split("_")
                
                # Check if all components satisfy the descendancy condition
                if check_complex_descendancy(net, source_components, target_components):
                    logger.debug(f"Adding inter-layer pair: {src_layer} -> {dst_layer} for source receptor: {source_receptor} and target ligand: {target_ligand}")
                    inter_layer_pairs.append((src_layer, dst_layer))
            completed_layers += 1
            if logger and (completed_layers % max(1, num_layers // 10) == 0):
                logger.info(f"Processed {completed_layers}/{num_layers} layers for inter-layer pairs")
        logger.info(f"Inter-layer pairs: {inter_layer_pairs} - {len(inter_layer_pairs)} pairs found")
    
    return inter_layer_pairs    

def get_expression_value_batch(data, gene_indexes, toll_complex, zero_threshold: float=1e-2) -> torch.Tensor:
    if len(gene_indexes) > 1:  # gene_id is a complex
        expr = torch.tensor(data[:, gene_indexes].X.astype(np.float32),
                            dtype=torch.float32)
        values = torch.exp(torch.mean(torch.log(expr + toll_complex), dim=1)).squeeze()
    else:
        values = torch.tensor(data[:, gene_indexes[0]].X.astype(np.float32),
                            dtype=torch.float32).squeeze()
        
    # Zero out small values
    values[values < zero_threshold] = 0.0
    return values

def compute_intralayer_interactions(data, dist_matrix, src_idx: int, 
                                  layer_src_info: Dict[str, Dict[str, List[int]]], 
                                  toll_complex: float) -> Tuple[torch.ShortTensor, torch.FloatTensor]:
    """
    Compute intralayer interactions for a specific layer.
    
    Args:
        data: Expression data matrix
        dist_matrix: Distance matrix between cells
        src_idx: Layer index
        layer_src_info: Dictionary containing ligand and receptor information
        toll_complex: Tolerance for complex computation
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - Indices tensor of shape (4, K) for K non-zero interactions
            - Values tensor of shape (K,) containing interaction strengths
    """
    # Extract gene information
    ligand_name = layer_src_info["ligand"]["gene_id"]
    ligand_indices = layer_src_info["ligand"]["component_indices"]
    receptor_name = layer_src_info["receptor"]["gene_id"]
    receptor_indices = layer_src_info["receptor"]["component_indices"]
    
    # Get expression values
    ligand_vals = get_expression_value_batch(data, ligand_indices, toll_complex)
    receptor_vals = get_expression_value_batch(data, receptor_indices, toll_complex)
    print(f"How many ligand zeros and receptor zeros: {torch.sum(ligand_vals == 0)} {torch.sum(receptor_vals == 0)}")
    # Compute interaction matrix
    interaction_matrix = torch.outer(ligand_vals, receptor_vals) / dist_matrix
    
    # Remove self-interactions
    interaction_matrix.fill_diagonal_(0)
    
    # Get non-zero positions and values
    nonzero_positions = torch.nonzero(interaction_matrix, as_tuple=False).to(torch.int16)
    
    # Handle empty case
    if nonzero_positions.numel() == 0:
        return torch.empty((4, 0), dtype=torch.long), torch.empty(0, dtype=torch.int16)
    
    # Extract values for non-zero positions
    nonzero_values = interaction_matrix[nonzero_positions[:, 0].to(torch.long), nonzero_positions[:, 1].to(torch.long)]
    
    # Create 4D indices [i, alpha, j, alpha]
    i_indices = nonzero_positions[:, 0]
    j_indices = nonzero_positions[:, 1]
    alpha_indices = torch.full_like(i_indices, src_idx)
    
    # Stack indices in correct order
    layer_indices = torch.stack([i_indices, alpha_indices, j_indices, alpha_indices])
    
    # Cleanup
    del interaction_matrix, ligand_vals, receptor_vals, nonzero_positions
    
    return layer_indices, nonzero_values

def compute_interlayer_interactions(data, dist_matrix, src_layer: int, dst_layer: int, src_info: Dict[str, List[int]], dst_info: Dict[str, List[int]], toll_complex: float) -> Tuple[torch.ShortTensor, torch.FloatTensor]:
    """
    Compute interlayer interactions between two layers.
    Handles all cases of nonzero positions (0-d tensor, 1-d tensor, or 2-d tensor).
    
    Args:
        data: Expression data
        dist_matrix: Distance matrix between cells
        src_layer: Source layer index
        dst_layer: Destination layer index
        src_info: Source layer gene information
        dst_info: Destination layer gene information
        toll_complex: Tolerance for complex computation
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Pair indices and values
    """
    receptor_src_name = src_info["receptor"]["gene_id"]
    receptor_src_indices = src_info["receptor"]["component_indices"]
    ligand_dst_name = dst_info["ligand"]["gene_id"]
    ligand_dst_indices = dst_info["ligand"]["component_indices"]
    receptor_vals = get_expression_value_batch(data, receptor_src_indices, toll_complex)
    ligand_vals = get_expression_value_batch(data, ligand_dst_indices, toll_complex)

     # Compute diagonal values
    diagonal_values = receptor_vals * ligand_vals
    
    # Get nonzero positions, handling all possible cases
    raw_nonzero = torch.nonzero(diagonal_values, as_tuple=False).to(torch.int16)
    
    # Handle empty case
    if raw_nonzero.numel() == 0:
        #print(f"No interactions found for src_layer {src_layer} and dst_layer {dst_layer}")
        return torch.empty((4, 0), dtype=torch.long), torch.empty(0, dtype=torch.int16)
    
    # Convert to 1D tensor regardless of input dimensionality
    if raw_nonzero.dim() == 0:
        # Single value case
        nonzero_positions = raw_nonzero.unsqueeze(0)
    elif raw_nonzero.dim() == 1:
        # Already 1D
        nonzero_positions = raw_nonzero
    else:
        # 2D case - flatten to 1D
        nonzero_positions = raw_nonzero.view(-1)
    
    # Get values for nonzero positions
    nonzero_values = diagonal_values[nonzero_positions.to(torch.long)]
    
    # Create indices for the sparse tensor
    i_indices = nonzero_positions
    src_indexes = torch.full_like(i_indices, src_layer)
    dst_indexes = torch.full_like(i_indices, dst_layer)
    
    # Stack as [dim0, dim1, dim2, dim3] = [i, alpha, i, beta]
    pair_indices = torch.stack([i_indices, src_indexes, i_indices, dst_indexes])
    
    # Cleanup
    del ligand_vals, receptor_vals, diagonal_values, raw_nonzero, nonzero_positions
    
    return pair_indices, nonzero_values
    
def compute_distance_matrix(cell_indexes, coord_x, coord_y, toll_distance=1e-6) -> torch.FloatTensor:
    """
    Compute a distance matrix for cells based on their coordinates.
    
    Args:
        cell_indexes: List of cell indices
        coord_x: List of x-coordinates for each cell
        coord_y: List of y-coordinates for each cell
        toll_distance: Distance threshold to consider interactions

    Returns:
        torch.FloatTensor: Distance matrix with shape (N, N)
    """
    N = len(cell_indexes)

    dist_matrix = torch.sqrt(
        (torch.tensor(coord_x).view(-1, 1) - torch.tensor(coord_x).view(1, -1)) ** 2 +
        (torch.tensor(coord_y).view(-1, 1) - torch.tensor(coord_y).view(1, -1)) ** 2
    ) + toll_distance  # Add tolerance to avoid division by zero
    dist_matrix = dist_matrix.fill_diagonal_(torch.inf)  # Remove self-interactions by setting diagonal to 0
    return dist_matrix

def get_layer_interaction(sparse_tensor, alpha, beta=None):
    """
    Extract interactions for specific layer(s) from sparse tensor.
    
    Args:
        sparse_tensor: torch.sparse.FloatTensor
        alpha: Source layer index
        beta: Target layer index (if None, returns intralayer interactions for alpha)
        
    Returns:
        torch.FloatTensor: 2D interaction matrix for the specified layer(s)
    """
    if beta is None:
        beta = alpha
    
    # Get indices and values
    indices = sparse_tensor.indices()
    values = sparse_tensor.values()
    
    # Find entries where dim1 == alpha and dim3 == beta
    mask = (indices[1] == alpha) & (indices[3] == beta)
    
    if not mask.any():
        # No interactions for this layer pair
        N = sparse_tensor.size(0)
        return torch.zeros(N, N, dtype=torch.float32)
    
    # Extract relevant indices and values
    relevant_indices = indices[:, mask]
    relevant_values = values[mask]
    
    # Create 2D sparse tensor
    N = sparse_tensor.size(0)
    layer_sparse = torch.sparse_coo_tensor(
        indices=relevant_indices[[0, 2]],  # [i, j] indices
        values=relevant_values,
        size=(N, N),
        dtype=torch.float32
    )
    
    return layer_sparse.to_dense()

def get_cell_interactions(sparse_tensor, cell_idx):
    """
    Extract all interactions involving a specific cell from sparse tensor.
    
    Args:
        sparse_tensor: torch.sparse.FloatTensor
        cell_idx: Cell index to extract interactions for
        
    Returns:
        dict: Dictionary with keys 'outgoing' and 'incoming' containing interaction matrices
    """
    indices = sparse_tensor.indices()
    values = sparse_tensor.values()
    
    L = sparse_tensor.size(1)
    
    # Outgoing interactions: cell_idx as source (dim0 == cell_idx)
    outgoing_mask = indices[0] == cell_idx
    outgoing_indices = indices[:, outgoing_mask]
    outgoing_values = values[outgoing_mask]
    
    # Incoming interactions: cell_idx as target (dim2 == cell_idx)
    incoming_mask = indices[2] == cell_idx
    incoming_indices = indices[:, incoming_mask]
    incoming_values = values[incoming_mask]
    
    # Create sparse matrices for outgoing [source_layer, target_cell, target_layer]
    if outgoing_mask.any():
        outgoing_sparse = torch.sparse_coo_tensor(
            indices=torch.stack([outgoing_indices[1], outgoing_indices[2], outgoing_indices[3]]),
            values=outgoing_values,
            size=(L, sparse_tensor.size(2), L),
            dtype=torch.float32
        )
    else:
        outgoing_sparse = torch.sparse_coo_tensor(
            indices=torch.zeros((3, 0), dtype=torch.long),
            values=torch.zeros(0, dtype=torch.float32),
            size=(L, sparse_tensor.size(2), L),
            dtype=torch.float32
        )
    
    # Create sparse matrices for incoming [source_cell, source_layer, target_layer]
    if incoming_mask.any():
        incoming_sparse = torch.sparse_coo_tensor(
            indices=torch.stack([incoming_indices[0], incoming_indices[1], incoming_indices[3]]),
            values=incoming_values,
            size=(sparse_tensor.size(0), L, L),
            dtype=torch.float32
        )
    else:
        incoming_sparse = torch.sparse_coo_tensor(
            indices=torch.zeros((3, 0), dtype=torch.long),
            values=torch.zeros(0, dtype=torch.float32),
            size=(sparse_tensor.size(0), L, L),
            dtype=torch.float32
        )
    
    return {
        'outgoing': outgoing_sparse,
        'incoming': incoming_sparse
    }

def select_lr_pairs(lrdb: pd.DataFrame, var_names: list[str], l: int):
    """
    Select number of ligand-receptor pairs from a DataFrame that are contained in dataset features.
    Args:
        lrdb: DataFrame containing ligand-receptor pairs with columns 'source' and 'target'
        var_names: List of variable names (genes) available in the dataset
        l: Number of pairs to select    
    Returns:
        DataFrame with selected ligand-receptor pairs, or None if not enough valid pairs are found
    """
        # Convert subdata.var_names to lowercase set for faster lookup
    valid_genes = set(g.lower() for g in var_names)

    # 2. Define a helper to check if all components of a complex are in valid_genes
    def is_valid_complex(gene_string):
        components = gene_string.split("_")
        return all(comp.lower() in valid_genes for comp in components)

    # 3. Apply filtering to both source and target
    filtered_df = lrdb[
    lrdb["source"].apply(is_valid_complex) &
       lrdb["target"].apply(is_valid_complex)
    ]
    # I need to transform only source and target columns to lowercase
    filtered_df[["source", "target"]] = filtered_df[["source", "target"]].apply(lambda x: x.str.lower())
    if len(filtered_df) >= l:
        sample_lr = filtered_df.sample(n=l, replace=False)
        notFound = False
    else:
        print(f"❌ Not enough valid interactions (found {len(filtered_df)}, needed {l})")
        sample_lr = None  # Or handle appropriately
    return sample_lr

if __name__ == "__main__":
    from mlnetst.core.knowledge.networks import load_resource
    import anndata as ad
    from pathlib import Path
    import logging

    class ColoredFormatter(logging.Formatter):
        """Custom formatter to add colors to log levels"""
        
        COLORS = {
            logging.DEBUG: "\033[94m",    # Blue
            logging.INFO: "\033[92m",     # Green
            logging.WARNING: "\033[93m",  # Yellow
            logging.ERROR: "\033[91m",    # Red
            logging.CRITICAL: "\033[95m", # Magenta
        }
        RESET = "\033[0m"  # Reset color
        
        def format(self, record):
            # Get the original formatted message
            message = super().format(record)
            # Add color based on log level
            color = self.COLORS.get(record.levelno, "")
            return f"{color}{message}{self.RESET}"

    # Set up colored logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    
    # Create colored formatter
    formatter = ColoredFormatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)

    net = load_resource("nichenet")
    print(net)

    x_hat_s = ad.read_h5ad(Path(__file__).parents[2] / "data" / "processed" / "mouse1_slice153_x_hat_s.h5ad")
    print(x_hat_s)

    lrdb = select_lr_pairs(
        net,
        var_names=x_hat_s.var_names.tolist(),
        l=5
    )
    logger.info(lrdb)
    ligand_ids = lrdb['source']
    receptor_ids = lrdb['target']
    layer_mapping = create_layer_gene_mapping(
        ligand_ids=ligand_ids.tolist(),
        receptor_ids=receptor_ids.tolist(),
        var_names=x_hat_s.var_names,
    )
    logger.info(layer_mapping)

    net_graph_version = nx.from_pandas_edgelist(
        net,
        source='source',
        target='target',
        edge_attr=['weight',"provenance"],
        create_using=nx.DiGraph
    )
    suitable_pairs = extract_suitable_layers_from_net(
        net=net_graph_version, 
        ligand_ids=ligand_ids.tolist(),
        receptor_ids=receptor_ids.tolist()
    )
    logger.info(suitable_pairs)
    inter_layer_pairs = select_inter_layers(
        suitable_pairs=suitable_pairs,
        layer_mapping=layer_mapping
    )
    logger.info(inter_layer_pairs)
