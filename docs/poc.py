import marimo

__generated_with = "0.13.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sys
    import time
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
    return Path, ad, load_resource, np, plt, time, torch


@app.cell
def _(Path, ad):
    x_hat_s = ad.read_h5ad(Path(__file__).parents[1] / "data" / "processed" / "mouse1_slice153_x_hat_s.h5ad")
    print(x_hat_s)
    return (x_hat_s,)


@app.cell
def _(np, x_hat_s):
    source, target = "Astro", "L2/3 IT"
    subdata = x_hat_s[x_hat_s.obs["subclass"].isin([source,target]), :]
    subdata.obsm["spatial"] = np.array([(x,y) for x,y in zip(subdata.obs["centroid_x"], subdata.obs["centroid_y"])])
    print(subdata)
    return (subdata,)


@app.cell
def _(load_resource, np, subdata, time, torch):
    def build_multilayer_network_fully_vectorized(n,l,data,interactions_df,toll_dist=1e-10,toll_geom_mean=1e-10,compute_intralayer=True,compute_interlayer=True,):
        """
        Memory-efficient fully vectorized multilayer network construction.

        Args:
            n, l: Network dimensions
            subdata, interactions_df: Data inputs
            toll_dist, toll_geom_mean: Numerical tolerances
            compute_intralayer: If True, compute same-layer interactions (default: True)
            compute_interlayer: If True, compute cross-layer interactions (default: True)
            n_jobs: Number of threads for parallel processing (default: 1)
                    -1 uses all available cores, 1 disables parallelization

        """
        tic = time.time()
        print(f"Intralyer: {'Yes' if compute_intralayer else 'No'}")
        print(f"Interlyer: {'Yes' if compute_interlayer else 'No'}")

        # Init Main tensor data structure
        mlnet = torch.zeros(n,l,n,l, dtype=torch.float32)

        # Extract the possible lr pairs
        notFound = True
        # Convert data.var_names to lowercase set for faster lookup
        valid_genes = set(g.lower() for g in data.var_names)

        # 2. Define a helper to check if all components of a complex are in valid_genes
        def is_valid_complex(gene_string):
            components = gene_string.split("_")
            return all(comp.lower() in valid_genes for comp in components)

        # 3. Apply filtering to both source and target
        filtered_df = interactions_df[
            interactions_df["source"].apply(is_valid_complex) &
            interactions_df["target"].apply(is_valid_complex)
        ]

        # 4. Sample from filtered interactions
        if len(filtered_df) >= l:
            sample_lr = filtered_df.sample(n=l)
            notFound = False
        else:
            print(f"❌ Not enough valid interactions (found {len(filtered_df)}, needed {l})")
            sample_lr = None  # Or handle appropriately

        # preextract IDS (minimal memory: 2L strings)
        ligand_ids = sample_lr["source"].str.lower().values
        receptor_ids = sample_lr["target"].str.lower().values

        # preextract cellindexes
        if N == len(data.obs_names):
            cell_indexes = data.obs_names
        else:
            cell_indexes = data.obs.sample(N, replace=False).index

        def get_expression_value(cell_id,gene_id):
            if len(gene_id.split("_")) > 1:
                gene_components = gene_id.split("_")
                expr = torch.tensor(data[cell_id, gene_components].X.astype(np.float32), dtype=torch.float32)
                result = torch.exp(torch.mean(torch.log(expr + toll_geom_mean), dim=1))
                del expr
                return result
            else:
                return torch.tensor(data[cell_id,gene_id].X.astype(np.float32))

        # Distance matrix computation for intralayer
        dist_matrix = None
        if compute_intralayer:
            spatial_position_tensor = torch.tensor(data[cell_indexes].obsm["spatial"], dtype=torch.float32)
            dist_matrix = torch.cdist(spatial_position_tensor, spatial_position_tensor, p=2)

            def compute_intralayer(alpha):
                """
                Compute intralayer interactions for a single layer
                """
                ligand_id = ligand_ids[alpha]
                receptor_id = receptor_ids[alpha]
                ligand_vals = get_expression_value(cell_indexes, ligand_id).squeeze()
                receptor_vals = get_expression_value(cell_indexes, receptor_id).squeeze()
                interaction_matrix = torch.div(torch.outer(ligand_vals, receptor_vals),dist_matrix + toll_dist)
                interaction_matrix.fill_diagonal_(0)
                return alpha, interaction_matrix

            for alpha in range(L):
                _, interaction_matrix = compute_intralayer(alpha)
                mlnet[:, alpha, :, alpha] = interaction_matrix
                del interaction_matrix

        if compute_interlayer:
            # Precompute all values
            ligand_matrix = torch.stack([
                get_expression_value(cell_indexes, lid).squeeze()
                for lid in ligand_ids
            ])  # shape: (L, N)

            receptor_matrix = torch.stack([
                get_expression_value(cell_indexes, rid).squeeze()
                for rid in receptor_ids
            ])  # shape: (L, N)

            # Vectorized interaction scores: (L, L, N)
            interlayer_matrix = receptor_matrix[:, None, :] * ligand_matrix[None, :, :]
            interlayer_matrix[torch.arange(L), torch.arange(L), :] = 0

            # Insert into mlnet diagonals
            for alpha in range(L):
                for beta in range(L):
                    # For each cell i: set mlnet[i, alpha, i, beta] = interaction score
                    mlnet[torch.arange(N), alpha, torch.arange(N), beta] = interlayer_matrix[alpha, beta]

        toc = time.time()
        print(f"Fully vectorized multilayer construction: {toc-tic:.2f} s")
        return mlnet

    def calculate_memory_requirements(n, l):
        """
        Calculate expected memory usage for multilayer network tensor.

        Args:
            N: Number of nodes
            L: Number of layers

        Returns:
            Memory requirements in bytes and human-readable format
        """

        # Main tensor: N × L × N × L elements
        total_elements = n * l * n * l  # This is N² × L²
        bytes_per_element = 4  # float32
        main_tensor_bytes = total_elements * bytes_per_element

        # Auxiliary memory (much smaller):
        # - Distance matrix: N × N × 4 bytes
        # - Temporary vectors: ~2N × 4 bytes per layer
        # - Coordinate vectors: 2N × 4 bytes
        auxiliary_bytes = n * n * 4 + 2 * n * 4 + 2 * n * 4

        total_bytes = main_tensor_bytes + auxiliary_bytes

        # Convert to human-readable format
        def bytes_to_human(bytes_val):
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if bytes_val < 1024:
                    return f"{bytes_val:.2f} {unit}"
                bytes_val /= 1024
            return f"{bytes_val:.2f} PB"

        print(f"Memory Analysis for N={n}, L={l}:")
        print(f"  Total elements in tensor: {total_elements:,}")
        print(f"  Main tensor memory: {bytes_to_human(main_tensor_bytes)}")
        print(f"  Auxiliary memory: {bytes_to_human(auxiliary_bytes)}")
        print(f"  Total expected memory: {bytes_to_human(total_bytes)}")
        print(f"  Main tensor dominates: {100 * main_tensor_bytes / total_bytes:.1f}% of total")

        return total_bytes

    N = len(subdata.obs_names)
    L = 10
    calculate_memory_requirements(N,L)
    lr_interactions_df = load_resource("mouseconsensus")
    mlnet = build_multilayer_network_fully_vectorized(N, L, subdata, lr_interactions_df)
    print(f"Num of non zero elements: {mlnet.count_nonzero()}, therefore the sparsity is: {(N*L*N*L - mlnet.count_nonzero())/(N*L*N*L) * 100:.2f}%")


    return (mlnet,)


@app.cell
def _(np, plt, time, torch):
    from scipy import stats
    from sklearn.mixture import GaussianMixture
    from scipy.optimize import minimize_scalar
    from scipy.signal import find_peaks
    import logging

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    def _elbow_method(weights, plot_diagnostics=False):
        """Find elbow point in sorted edge weights"""
        start_time = time.time()
    
        sorted_weights = np.sort(weights)[::-1]  # Descending order
        n_points = len(sorted_weights)
        x = np.arange(n_points)
    
        # Normalize for distance calculation
        normalized_x = x / n_points
        normalized_y = (sorted_weights - sorted_weights.min()) / (sorted_weights.max() - sorted_weights.min())
    
        # Distance from line connecting first and last point
        distances = []
        for i in range(1, len(normalized_x)-1):
            # Distance from point to line
            x1, y1 = 0, normalized_y[0]
            x2, y2 = 1, normalized_y[-1]
            x0, y0 = normalized_x[i], normalized_y[i]
        
            dist = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2)
            distances.append(dist)
    
        elbow_idx = np.argmax(distances) + 1
        threshold = sorted_weights[elbow_idx]
    
        logger.info(f"Elbow method took {time.time() - start_time:.3f}s")
    
        if plot_diagnostics:
            plt.subplot(2, 3, 1)
            plt.plot(sorted_weights, 'b-', alpha=0.7)
            plt.axvline(elbow_idx, color='r', linestyle='--', label=f'Elbow at {elbow_idx}')
            plt.axhline(threshold, color='r', linestyle=':', label=f'Threshold: {threshold:.4f}')
            plt.xlabel('Edge rank')
            plt.ylabel('Edge weight')
            plt.title('Elbow Method')
            plt.legend()
            plt.yscale('log')
    
        return threshold

    def _zscore_method(weights, plot_diagnostics=False, z_threshold=3.0):
        """Select edges using z-score threshold"""
        start_time = time.time()
    
        mean_w = np.mean(weights)
        std_w = np.std(weights)
        threshold = mean_w + z_threshold * std_w
    
        logger.info(f"Z-score method took {time.time() - start_time:.3f}s")
    
        if plot_diagnostics:
            plt.subplot(2, 3, 2)
            plt.hist(weights, bins=100, alpha=0.7, density=True)
            plt.axvline(threshold, color='r', linestyle='--', label=f'Z-score threshold: {threshold:.4f}')
            plt.axvline(mean_w, color='g', linestyle=':', alpha=0.7, label=f'Mean: {mean_w:.4f}')
            plt.xlabel('Edge weight')
            plt.ylabel('Density')
            plt.title('Z-Score Method')
            plt.legend()
            plt.yscale('log')
    
        return threshold

    def _mad_method(weights, plot_diagnostics=False, mad_threshold=3.0):
        """Select edges using Median Absolute Deviation"""
        start_time = time.time()
    
        median_w = np.median(weights)
        mad = np.median(np.abs(weights - median_w))
        threshold = median_w + mad_threshold * mad
    
        logger.info(f"MAD method took {time.time() - start_time:.3f}s")
    
        if plot_diagnostics:
            plt.subplot(2, 3, 3)
            plt.hist(weights, bins=100, alpha=0.7, density=True)
            plt.axvline(threshold, color='r', linestyle='--', label=f'MAD threshold: {threshold:.4f}')
            plt.axvline(median_w, color='g', linestyle=':', alpha=0.7, label=f'Median: {median_w:.4f}')
            plt.xlabel('Edge weight')
            plt.ylabel('Density')
            plt.title('MAD Method')
            plt.legend()
            plt.yscale('log')
    
        return threshold

    def _gmm_method(weights, plot_diagnostics=False, n_components=2):
        """Use Gaussian Mixture Model to separate noise from signal"""
        start_time = time.time()
    
        weights_reshaped = weights.reshape(-1, 1)
    
        # Fit GMM
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(weights_reshaped)
    
        # Get the component with higher mean as "signal"
        means = gmm.means_.flatten()
        signal_component = np.argmax(means)
    
        # Set threshold at intersection of gaussians or at mean + 2*std of noise component
        noise_component = 1 - signal_component
        noise_mean = means[noise_component]
        noise_std = np.sqrt(gmm.covariances_[noise_component, 0, 0])
    
        threshold = noise_mean + 2 * noise_std
    
        logger.info(f"GMM method took {time.time() - start_time:.3f}s")
    
        if plot_diagnostics:
            plt.subplot(2, 3, 4)
            plt.hist(weights, bins=100, alpha=0.7, density=True)
        
            # Plot GMM components
            x_range = np.linspace(weights.min(), weights.max(), 1000)
            for i in range(n_components):
                component_pdf = stats.norm.pdf(x_range, means[i], np.sqrt(gmm.covariances_[i, 0, 0]))
                plt.plot(x_range, gmm.weights_[i] * component_pdf, 
                        label=f'Component {i} (μ={means[i]:.4f})')
        
            plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
            plt.xlabel('Edge weight')
            plt.ylabel('Density')
            plt.title('GMM Method')
            plt.legend()
            plt.yscale('log')
    
        return threshold

    def _otsu_method(weights, plot_diagnostics=False):
        """Use Otsu's method for automatic thresholding"""
        start_time = time.time()
    
        # Create histogram
        hist, bin_edges = np.histogram(weights, bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
        # Otsu's method
        total = len(weights)
        current_max = 0
        threshold = 0
    
        sum_total = np.sum(bin_centers * hist)
        sum_background = 0
        weight_background = 0
    
        for i in range(len(hist)):
            weight_background += hist[i]
            if weight_background == 0:
                continue
            
            weight_foreground = total - weight_background
            if weight_foreground == 0:
                break
            
            sum_background += bin_centers[i] * hist[i]
        
            mean_background = sum_background / weight_background
            mean_foreground = (sum_total - sum_background) / weight_foreground
        
            # Between-class variance
            variance_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        
            if variance_between > current_max:
                current_max = variance_between
                threshold = bin_centers[i]
    
        logger.info(f"Otsu method took {time.time() - start_time:.3f}s")
    
        if plot_diagnostics:
            plt.subplot(2, 3, 5)
            plt.hist(weights, bins=100, alpha=0.7, density=True)
            plt.axvline(threshold, color='r', linestyle='--', label=f'Otsu threshold: {threshold:.4f}')
            plt.xlabel('Edge weight')
            plt.ylabel('Density')
            plt.title('Otsu Method')
            plt.legend()
            plt.yscale('log')
    
        return threshold

    def _histogram_valley_method(weights, plot_diagnostics=False):
        """Find valley in histogram as threshold"""
        start_time = time.time()
    
        hist, bin_edges = np.histogram(weights, bins=100)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
        # Smooth histogram
        from scipy.ndimage import gaussian_filter1d
        smooth_hist = gaussian_filter1d(hist.astype(float), sigma=1)
    
        # Find peaks and valleys
        peaks, _ = find_peaks(smooth_hist)
        valleys, _ = find_peaks(-smooth_hist)
    
        if len(valleys) > 0:
            # Choose the valley with highest bin_center value (rightmost valley)
            valley_idx = valleys[np.argmax(bin_centers[valleys])]
            threshold = bin_centers[valley_idx]
        else:
            # Fallback to median + 2*MAD
            threshold = np.median(weights) + 2 * np.median(np.abs(weights - np.median(weights)))
    
        logger.info(f"Histogram valley method took {time.time() - start_time:.3f}s")
    
        if plot_diagnostics:
            plt.subplot(2, 3, 6)
            plt.plot(bin_centers, hist, 'b-', alpha=0.7, label='Histogram')
            plt.plot(bin_centers, smooth_hist, 'g-', label='Smoothed')
            if len(valleys) > 0:
                plt.scatter(bin_centers[valleys], smooth_hist[valleys], 
                           color='orange', s=50, label='Valleys')
            plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
            plt.xlabel('Edge weight')
            plt.ylabel('Count')
            plt.title('Histogram Valley Method')
            plt.legend()
            plt.yscale('log')
    
        return threshold

    def select_edges_by_method(edge_weights, method='elbow', plot_diagnostics=False):
        """
        Select edges using various adaptive thresholding methods.
    
        Parameters:
        -----------
        edge_weights : torch.Tensor or np.array
            Flattened array of edge weights
        method : str
            Method to use: 'elbow', 'zscore', 'mad', 'gmm', 'otsu', 'histogram_valley'
        plot_diagnostics : bool
            Whether to plot diagnostic information
        
        Returns:
        --------
        threshold : float
            Selected threshold value
        selected_edges : bool array
            Boolean mask for selected edges
        """
    
        start_time = time.time()
        logger.info(f"Starting edge selection with method: {method}")
    
        # Convert to numpy and flatten
        if isinstance(edge_weights, torch.Tensor):
            weights = edge_weights.detach().cpu().numpy().flatten()
        else:
            weights = edge_weights.flatten()
    
        # Remove zeros/very small values that might be noise
        original_size = len(weights)
        weights = weights[weights > 1e-10]
        logger.info(f"Filtered {original_size - len(weights)} near-zero weights, {len(weights)} remaining")
    
        if plot_diagnostics:
            plt.figure(figsize=(15, 10))
    
        if method == 'elbow':
            threshold = _elbow_method(weights, plot_diagnostics)
        elif method == 'zscore':
            threshold = _zscore_method(weights, plot_diagnostics)
        elif method == 'mad':
            threshold = _mad_method(weights, plot_diagnostics)
        elif method == 'gmm':
            threshold = _gmm_method(weights, plot_diagnostics)
        elif method == 'otsu':
            threshold = _otsu_method(weights, plot_diagnostics)
        elif method == 'histogram_valley':
            threshold = _histogram_valley_method(weights, plot_diagnostics)
        else:
            raise ValueError(f"Unknown method: {method}")
    
        selected_edges = edge_weights > threshold
        n_selected = selected_edges.sum()
    
        total_time = time.time() - start_time
        logger.info(f"Edge selection completed in {total_time:.3f}s")
        logger.info(f"Method: {method}")
        logger.info(f"Threshold: {threshold:.6f}")
        logger.info(f"Selected edges: {n_selected} ({n_selected/len(weights)*100:.2f}%)")
    
        if plot_diagnostics:
            plt.suptitle(f'Edge Selection Diagnostics - {method.upper()}', fontsize=16)
            plt.tight_layout()
            plt.show()
    
        return threshold, selected_edges
    return logger, select_edges_by_method


@app.cell
def _(logger, np, plt, select_edges_by_method, time, torch):
    import math
    def apply_to_network_data(mlnet, subdata, method='elbow', num_layer_to_plot=3, draw_intralayer_edges = False, draw_interlayer_edges = False, show_diagnostics=True):
        """
        Apply edge selection to your network data with comprehensive timing
    
        Parameters:
        -----------
        mlnet : torch.Tensor
            Multi-layer network tensor [nodes, layers, nodes, layers]
        subdata : object
            Data object with .obs containing 'centroid_x' and 'centroid_y'
        method : str
            Thresholding method to use
        show_diagnostics : bool
            Whether to show diagnostic plots
        """


        total_start_time = time.time()
        logger.info("=" * 60)
        logger.info(f"Starting network visualization with method: {method}\n\tDraw Intra: {draw_intralayer_edges}\n\tDraw Inter: {draw_interlayer_edges}")
        logger.info(f"Network shape: {mlnet.shape}")
        mlnet = mlnet[:,:num_layer_to_plot,:,:num_layer_to_plot]
        logger.info(f"Shape of the mlnet to draw: {mlnet.shape}")
    
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(projection="3d")
    
        total_edges_selected = 0
        total_nodes = 0

        def map_label_to_color(label):
            if label == "Astro":
                return "red"
            else:
                return "blue"

        symbol_list = ['o'] * num_layer_to_plot
        startz, endz = -50, 50
        zdepth_list = torch.arange(start = startz, end = endz, step = (endz - startz) // num_layer_to_plot)
        graphical_parameters_layers = [(symbol, zdepth) for symbol, zdepth in zip(symbol_list, zdepth_list)]
        for ilayer, (m, z) in enumerate(graphical_parameters_layers):
            layer_start_time = time.time()
            logger.info(f"\n--- Processing Layer {ilayer} ---")
        
            # Extract intralayer edges
            if draw_intralayer_edges:
                intralayer_edges = mlnet[:, ilayer, :, ilayer].squeeze()
                logger.info(f"Layer {ilayer} edge matrix shape: {intralayer_edges.shape}")
                logger.info(f"Layer {ilayer} edge weight range: [{intralayer_edges.min():.6f}, {intralayer_edges.max():.6f}]")
            
                # Apply intelligent edge selection
                threshold_start_time = time.time()
                threshold, selected_edges = select_edges_by_method(
                    intralayer_edges, 
                    method=method, 
                    plot_diagnostics=(show_diagnostics and ilayer == 0)  # Show diagnostics for first layer only
                )
                threshold_time = time.time() - threshold_start_time
            
                # Apply threshold
                filtered_edges = intralayer_edges * selected_edges
            
                # Get edge indices
                edge_indices = torch.argwhere(filtered_edges)
                n_edges = edge_indices.shape[0]
                total_edges_selected += n_edges
            
                logger.info(f"Layer {ilayer}: {n_edges} edges selected")
                if n_edges == 0:
                    logger.warning(f"No edges found for layer {ilayer}, skipping visualization")
                    continue
            else:
                logger.info(f"Not drawing intralayer edges...")
                n_edges = 0
                threshold_time = 0
        
            # Get all node positions for this layer
            plotting_start_time = time.time()
            num_nodes = subdata.shape[0]
            total_nodes = num_nodes  # Assuming same for all layers
        
            xs = subdata.obs["centroid_x"].values
            zs = subdata.obs["centroid_y"].values
            ys = np.full(num_nodes, z)
        
            logger.info(f"Layer {ilayer}: plotting {num_nodes} nodes and {n_edges} edges")
        
            # Plot all nodes as scatter points
            scatter_start = time.time()
        
            map_label_to_color = {"Astro": "red", "L2/3 IT": "blue"}
            color_nodes = [map_label_to_color[x] for x in subdata.obs["subclass"].values]
            ax.scatter(xs, ys, zs, marker=m, s=50, alpha=0.7, label=f'Layer {ilayer}', c=color_nodes)
            scatter_time = time.time() - scatter_start
        
            # Plot edges (lines connecting source to target)
            intralines_start = time.time()
        
            # Vectorized approach for better performance
            if draw_intralayer_edges:
                if n_edges > 0:
                    source_indices = edge_indices[:, 0].cpu().numpy()
                    target_indices = edge_indices[:, 1].cpu().numpy()
                
                    # Batch create line coordinates
                    x_coords = np.column_stack([xs[source_indices], xs[target_indices]])
                    y_coords = np.column_stack([np.full(n_edges, z), np.full(n_edges, z)])
                    z_coords = np.column_stack([zs[source_indices], zs[target_indices]])
                
                    # Plot all lines at once (more efficient)
                    for i in range(n_edges):
                        ax.plot([x_coords[i, 0], x_coords[i, 1]], 
                               [y_coords[i, 0], y_coords[i, 1]], 
                               [z_coords[i, 0], z_coords[i, 1]], 
                               linewidth=0.1, color="grey", alpha=0.5)
        
            lines_time = time.time() - intralines_start
            plotting_time = time.time() - plotting_start_time
            layer_time = time.time() - layer_start_time
        
            logger.info(f"Layer {ilayer} timing breakdown:")
            logger.info(f"  - Thresholding: {threshold_time:.3f}s")
            logger.info(f"  - Scatter plot: {scatter_time:.3f}s")
            logger.info(f"  - Line plotting: {lines_time:.3f}s")
            logger.info(f"  - Total plotting: {plotting_time:.3f}s")
            logger.info(f"  - Total layer time: {layer_time:.3f}s")

        if draw_interlayer_edges:
            inter_layer_start_time = time.time()
            cmap = plt.get_cmap("tab10")
    
            for inode in range(num_nodes):
                interlayer_edges = mlnet[inode, :, inode, :].squeeze()
                interlayer_edges = interlayer_edges * (interlayer_edges > np.quantile(interlayer_edges, 0.99))
                ids_layers = interlayer_edges.argwhere()
    
                n_inter_edges = ids_layers.shape[0]
                for i in range(n_inter_edges):
                    layer1_idx = ids_layers[i,0]
                    layer2_idx = ids_layers[i,1]
    
                    # Use color based on the distance between layers
                    distance = abs(layer2_idx - layer1_idx)
                    color_of_edge = cmap(distance % 10)  # Tab10 has 10 distinct colors
    
                    # Same x,z coordinates (same node), different y coordinates (different layers)
                    x_inter_coords = [xs[inode], xs[inode]]
                    y_inter_coords = [zdepth_list[layer1_idx], zdepth_list[layer2_idx]]
                    z_inter_coords = [zs[inode], zs[inode]]
    
                    ax.plot(
                        x_inter_coords, y_inter_coords, z_inter_coords,
                        linewidth=1, color=color_of_edge, alpha=0.5
                    )
                
            finalize_inter_layer = time.time() - inter_layer_start_time
        else:
            finalize_inter_layer = 0
        
        # Finalize plot
        finalize_start = time.time()
        ax.set_xlabel('X (centroid_x)')
        ax.set_ylabel('Y (layer)')
        ax.set_zlabel('Z (centroid_y)')
        ax.legend()
        plt.title(f'3D Network Visualization - {method.upper()} method')
        finalize_time = time.time() - finalize_start
    
        total_time = time.time() - total_start_time
    
        # Summary statistics
        logger.info("\n" + "=" * 60)
        logger.info("VISUALIZATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total nodes: {total_nodes}")
        logger.info(f"Total edges selected: {total_edges_selected}")
        logger.info(f"Selection rate: {total_edges_selected/(total_nodes**2 * 3)*100:.3f}%")
        logger.info(f"Inter layer plot: {finalize_inter_layer:.3f}s")
        logger.info(f"Plot finalization: {finalize_time:.3f}s")
        logger.info(f"TOTAL TIME: {total_time:.3f}s")
        logger.info("=" * 60)

        return fig, ax

    return (apply_to_network_data,)


@app.cell
def _(apply_to_network_data, mlnet, plt, subdata):
    fig, ax = apply_to_network_data(mlnet,subdata,method="elbow", num_layer_to_plot=3, draw_intralayer_edges = True, draw_interlayer_edges = True, show_diagnostics=False)
    #ax.view_init(azim=180)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
