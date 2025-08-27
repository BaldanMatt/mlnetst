
class NetworkAnalyzer:
    """
    Enhanced network analyzer with parameterizable plotting options.
    
    This class provides flexible visualization methods for multilayer network analysis
    with customizable plot parameters, labels, colors, and styling options.
    """
    
    def __init__(self, 
                 experiment_name: str, 
                 logger: logging.Logger,
                 media_dir: Optional[Path] = None,
                 default_figsize: Tuple[int, int] = (12, 8),
                 default_dpi: int = 300,
                 default_style: str = "whitegrid"):
        """
        Initialize the NetworkAnalyzer with customizable defaults.
        
        Args:
            experiment_name: Name for the experiment (used in file naming)
            logger: Logger instance for output messages
            media_dir: Directory to save figures (if None, uses default)
            default_figsize: Default figure size as (width, height)
            default_dpi: Default DPI for saved figures
            default_style: Default seaborn style
        """
        self.experiment_name = experiment_name
        self.logger = logger
        self.default_figsize = default_figsize
        self.default_dpi = default_dpi
        
        # Set up media directory
        if media_dir is None:
            project_root = Path(__file__).parents[1]
            self.media_dir = project_root / "media"
        else:
            self.media_dir = Path(media_dir)
        
        self.media_dir.mkdir(exist_ok=True)
        
        # Set seaborn style
        sns.set_style(default_style)
        
    def _save_figure(self, 
                    fig_name: str, 
                    dpi: Optional[int] = None,
                    bbox_inches: str = 'tight',
                    format: str = 'png') -> Path:
        """
        Save figure with consistent naming and parameters.
        
        Args:
            fig_name: Base name for the figure
            dpi: DPI for saved figure (uses default if None)
            bbox_inches: Bounding box specification
            format: File format for saving
            
        Returns:
            Path to saved figure
        """
        dpi = dpi or self.default_dpi
        filename = f"{self.experiment_name}_{fig_name}.{format}"
        filepath = self.media_dir / filename
        
        plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, format=format)
        plt.close()
        self.logger.info(f"Figure saved: {filepath}")
        return filepath
    
    def plot_degree_distributions(self, 
                                  metrics_data: Dict[str, torch.Tensor],
                                  plot_config: Optional[Dict[str, Any]] = None) -> List[Path]:
        """
        Create customizable degree distribution plots.
        
        Args:
            metrics_data: Dictionary mapping metric names to tensor values
            plot_config: Configuration dictionary for plot customization
            
        Returns:
            List of paths to saved figures
        """
        # Default configuration
        default_config = {
            'figsize': self.default_figsize,
            'violin_colors': None,  # Use seaborn defaults
            'hist_colors': ['skyblue', 'lightcoral', 'lightgreen', 'gold'],
            'alpha': 0.7,
            'bins': 30,
            'edge_color': 'black',
            'title_fontsize': 14,
            'label_fontsize': 12,
            'violin_title': "Distribution of Network Metrics",
            'hist_title_template': "{metric} Distribution",
            'xlabel': "Metric Type",
            'ylabel': "Value",
            'hist_xlabel_template': "{metric}",
            'hist_ylabel': "Frequency"
        }
        
        # Update with user config
        config = {**default_config, **(plot_config or {})}
        
        self.logger.info("Creating degree distribution plots")
        
        # Prepare data for violin plots
        values = []
        metric_names = []
        for name, tensor in metrics_data.items():
            values.extend(tensor.cpu().numpy())
            metric_names.extend([name] * len(tensor))
        
        df_metrics = pd.DataFrame({
            "value": values,
            "metric": metric_names
        })
        
        # Create violin plot
        plt.figure(figsize=config['figsize'])
        violin_plot = sns.violinplot(data=df_metrics, x="metric", y="value", 
                                   palette=config['violin_colors'])
        plt.title(config['violin_title'], fontsize=config['title_fontsize'])
        plt.ylabel(config['ylabel'], fontsize=config['label_fontsize'])
        plt.xlabel(config['xlabel'], fontsize=config['label_fontsize'])
        plt.xticks(rotation=45)
        violin_path = self._save_figure("degree_distributions_violin")
        
        # Create histogram comparison
        num_metrics = len(metrics_data)
        cols = min(3, num_metrics)  # Max 3 columns
        rows = (num_metrics + cols - 1) // cols  # Ceiling division
        
        fig, axes = plt.subplots(rows, cols, figsize=(config['figsize'][0] * cols / 2, 
                                                     config['figsize'][1] * rows / 2))
        if num_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
        
        colors = config['hist_colors']
        
        for idx, (metric_name, tensor) in enumerate(metrics_data.items()):
            if idx < len(axes):
                ax = axes[idx]
                color = colors[idx % len(colors)]
                
                ax.hist(tensor.cpu().numpy(), 
                       bins=config['bins'], 
                       alpha=config['alpha'], 
                       color=color, 
                       edgecolor=config['edge_color'])
                
                ax.set_title(config['hist_title_template'].format(metric=metric_name.title()),
                           fontsize=config['title_fontsize'])
                ax.set_xlabel(config['hist_xlabel_template'].format(metric=metric_name.title()),
                            fontsize=config['label_fontsize'])
                ax.set_ylabel(config['hist_ylabel'], fontsize=config['label_fontsize'])
        
        # Hide extra subplots
        for idx in range(len(metrics_data), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        hist_path = self._save_figure("degree_distributions_histogram")
        
        return [violin_path, hist_path]
    
    def plot_spatial_networks(self, 
                            subdata: anndata.AnnData,
                            cell_indexes: List[str],
                            metrics_data: Dict[str, torch.Tensor],
                            plot_config: Optional[Dict[str, Any]] = None,
                            figname: str = "spatial_analysis") -> List[Path]:
        """
        Create enhanced spatial network visualizations.
        
        Args:
            subdata: AnnData object with spatial information
            cell_indexes: List of cell identifiers
            metrics_data: Dictionary mapping metric names to tensor values
            plot_config: Configuration dictionary for plot customization
            figname: Base name for the output figure
            
        Returns:
            List of paths to saved figures
        """
        # Default configuration
        default_config = {
            'figsize': (20, 16),
            'scatter_size': 25,  # Reduced default size
            'scatter_alpha': 0.6,
            'correlation_scatter_size': 20,  # Even smaller for correlation plots
            'correlation_alpha': 0.5,
            'colormaps': ['viridis', 'plasma', 'cividis', 'magma'],
            'coord_columns': ['centroid_x', 'centroid_y'],
            'xlabel': "X Coordinate",
            'ylabel': "Y Coordinate",
            'title_fontsize': 14,
            'label_fontsize': 12,
            'colorbar_label_fontsize': 10,
            'trend_line_color': 'red',
            'trend_line_style': '--',
            'trend_line_alpha': 0.8,
            'correlation_precision': 3,
            'subplot_titles': None,  # Auto-generate if None
            'show_correlation_matrix': True,
            'combined_view_markers': ['o', 's', '^', 'D'],
            'combined_view_sizes': [60, 40, 45, 35]
        }
        
        # Update with user config
        config = {**default_config, **(plot_config or {})}
        
        self.logger.info("Creating spatial network plots")
        
        # Create graph with spatial positions and metrics
        g = nx.Graph()
        coord_x, coord_y = config['coord_columns']
        
        nodes_dict = {}
        for idx, cell_id in enumerate(cell_indexes):
            node_data = {
                "x": subdata.obs.loc[cell_id, coord_x],
                "y": subdata.obs.loc[cell_id, coord_y]
            }
            # Add all metrics to node data
            for metric_name, tensor in metrics_data.items():
                node_data[metric_name] = float(tensor[idx])
            nodes_dict[cell_id] = node_data
        
        g.add_nodes_from(nodes_dict.items())
        
        # Extract positions and metric values
        pos = {cell_id: (nodes_dict[cell_id]["x"], nodes_dict[cell_id]["y"]) 
               for cell_id in nodes_dict}
        
        metric_values = {}
        for metric_name in metrics_data.keys():
            metric_values[metric_name] = [nodes_dict[cell_id][metric_name] 
                                        for cell_id in nodes_dict]
        
        # Log statistics
        self._log_network_stats(metric_values)
        
        # Determine subplot layout
        num_metrics = len(metrics_data)
        if config['show_correlation_matrix'] and num_metrics > 1:
            # Include correlation plots
            total_plots = num_metrics + min(3, num_metrics * (num_metrics - 1) // 2)
        else:
            total_plots = num_metrics
        
        # Calculate optimal subplot layout
        cols = min(3, total_plots)
        rows = (total_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=config['figsize'])
        if total_plots == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
        
        plot_idx = 0
        colormaps = config['colormaps']
        
        # Individual metric spatial plots
        for idx, (metric_name, values) in enumerate(metric_values.items()):
            if plot_idx < len(axes):
                ax = axes[plot_idx]
                cmap = colormaps[idx % len(colormaps)]
                
                scatter = ax.scatter([pos[cell_id][0] for cell_id in nodes_dict], 
                                   [pos[cell_id][1] for cell_id in nodes_dict],
                                   c=values, 
                                   cmap=cmap, 
                                   s=config['scatter_size'], 
                                   alpha=config['scatter_alpha'])
                
                title = (config['subplot_titles'][idx] if config['subplot_titles'] 
                        else f"Spatial {metric_name.replace('_', ' ').title()} Distribution")
                ax.set_title(title, fontsize=config['title_fontsize'])
                ax.set_xlabel(config['xlabel'], fontsize=config['label_fontsize'])
                ax.set_ylabel(config['ylabel'], fontsize=config['label_fontsize'])
                
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(metric_name.replace('_', ' ').title(), 
                              fontsize=config['colorbar_label_fontsize'])
                
                plot_idx += 1
        
        # Correlation plots (if enabled and multiple metrics)
        if config['show_correlation_matrix'] and len(metric_values) > 1:
            metric_names = list(metric_values.keys())
            correlation_count = 0
            max_correlations = min(3, len(axes) - plot_idx)  # Limit to available space
            
            for i in range(len(metric_names)):
                for j in range(i + 1, len(metric_names)):
                    if correlation_count >= max_correlations or plot_idx >= len(axes):
                        break
                    
                    ax = axes[plot_idx]
                    x_vals = metric_values[metric_names[i]]
                    y_vals = metric_values[metric_names[j]]
                    
                    correlation = np.corrcoef(x_vals, y_vals)[0, 1]
                    
                    ax.scatter(x_vals, y_vals, 
                             alpha=config['correlation_alpha'], 
                             s=config['correlation_scatter_size'])
                    
                    # Add trend line
                    z = np.polyfit(x_vals, y_vals, 1)
                    p = np.poly1d(z)
                    ax.plot(x_vals, p(x_vals), 
                           color=config['trend_line_color'],
                           linestyle=config['trend_line_style'], 
                           alpha=config['trend_line_alpha'])
                    
                    ax.set_xlabel(metric_names[i].replace('_', ' ').title(),
                                fontsize=config['label_fontsize'])
                    ax.set_ylabel(metric_names[j].replace('_', ' ').title(),
                                fontsize=config['label_fontsize'])
                    
                    correlation_str = f"r={correlation:.{config['correlation_precision']}f}"
                    title = f"{metric_names[i].title()} vs {metric_names[j].title()} ({correlation_str})"
                    ax.set_title(title, fontsize=config['title_fontsize'])
                    
                    plot_idx += 1
                    correlation_count += 1
                
                if correlation_count >= max_correlations:
                    break
        
        # Hide extra subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        spatial_path = self._save_figure(f"spatial_network_analysis_{figname}")
        
        return [spatial_path]
    
    def plot_correlation_matrix(self,
                               metrics_data: Dict[str, torch.Tensor],
                               plot_config: Optional[Dict[str, Any]] = None) -> List[Path]:
        """
        Create a comprehensive correlation matrix heatmap.
        
        Args:
            metrics_data: Dictionary mapping metric names to tensor values
            plot_config: Configuration dictionary for plot customization
            
        Returns:
            List of paths to saved figures
        """
        default_config = {
            'figsize': (10, 8),
            'cmap': 'coolwarm',
            'center': 0,
            'annot': True,
            'fmt': '.3f',
            'title': 'Network Metrics Correlation Matrix',
            'title_fontsize': 16,
            'annot_fontsize': 10
        }
        
        config = {**default_config, **(plot_config or {})}
        
        # Create correlation matrix
        df_data = {}
        for name, tensor in metrics_data.items():
            df_data[name.replace('_', ' ').title()] = tensor.cpu().numpy()
        
        df = pd.DataFrame(df_data)
        correlation_matrix = df.corr()
        
        # Create heatmap
        plt.figure(figsize=config['figsize'])
        sns.heatmap(correlation_matrix, 
                   cmap=config['cmap'],
                   center=config['center'],
                   annot=config['annot'],
                   fmt=config['fmt'],
                   square=True,
                   annot_kws={'fontsize': config['annot_fontsize']})
        
        plt.title(config['title'], fontsize=config['title_fontsize'])
        plt.tight_layout()
        
        correlation_path = self._save_figure("correlation_matrix")
        return [correlation_path]
    
    def create_custom_plot(self,
                          plot_func: callable,
                          plot_args: tuple = (),
                          plot_kwargs: Optional[Dict[str, Any]] = None,
                          fig_name: str = "custom_plot",
                          figsize: Optional[Tuple[int, int]] = None) -> Path:
        """
        Create a custom plot using a user-defined function.
        
        Args:
            plot_func: Function that creates the plot
            plot_args: Positional arguments for plot_func
            plot_kwargs: Keyword arguments for plot_func
            fig_name: Name for the saved figure
            figsize: Figure size (uses default if None)
            
        Returns:
            Path to saved figure
        """
        figsize = figsize or self.default_figsize
        plot_kwargs = plot_kwargs or {}
        
        plt.figure(figsize=figsize)
        plot_func(*plot_args, **plot_kwargs)
        
        custom_path = self._save_figure(fig_name)
        return custom_path
    
    def _log_network_stats(self, metric_values: Dict[str, List[float]]):
        """
        Log comprehensive network statistics.
        
        Args:
            metric_values: Dictionary mapping metric names to value lists
        """
        self.logger.info("Network Statistics Summary:")
        self.logger.info("-" * 50)
        
        for metric_name, values in metric_values.items():
            values_array = np.array(values)
            stats = {
                'Min': np.min(values_array),
                'Max': np.max(values_array),
                'Mean': np.mean(values_array),
                'Median': np.median(values_array),
                'Std': np.std(values_array),
                'Q1': np.percentile(values_array, 25),
                'Q3': np.percentile(values_array, 75)
            }
            
            self.logger.info(f"{metric_name.upper()}:")
            for stat_name, stat_value in stats.items():
                self.logger.info(f"  {stat_name}: {stat_value:.3f}")
            self.logger.info("")


# Example usage function to replace the original analyze_and_visualize_network
def analyze_and_visualize_network_enhanced(
    mlnet: torch.Tensor,
    subdata: anndata.AnnData,
    cell_indexes: List[str],
    args: argparse.Namespace,
    logger: logging.Logger
) -> None:
    """
    Enhanced network analysis with improved NetworkAnalyzer.
    
    This replaces the original analyze_and_visualize_network function
    with more flexible plotting options.
    """
    logger.info("Starting enhanced network analysis and visualization")
    
    # Compute network metrics (same as before)
    logger.info("Computing network metrics")
    num_layers = mlnet.shape[1]
    num_cells = mlnet.shape[0]
    
    # Import required functions (you'll need to ensure these are available)
    from mlnetst.utils.mlnet_utils import build_supra_adjacency_matrix_from_tensor
    from mlnetst.utils.mlnet_metrics_utils import (
        compute_instrength, compute_multi_instrength,
        compute_outstrength, compute_multi_outstrength,
        compute_indegree, compute_multi_indegree,
        compute_outdegree, compute_multi_outdegree
    )
    
    supra_adjacency = build_supra_adjacency_matrix_from_tensor(mlnet)
    
    # Compute all metrics
    metrics_data = {
        'in_strength': compute_instrength(supra_adjacency, num_cells, num_layers),
        'out_strength': compute_outstrength(supra_adjacency, num_cells, num_layers),
        'in_degree': compute_indegree(supra_adjacency, num_cells, num_layers),
        'out_degree': compute_outdegree(supra_adjacency, num_cells, num_layers),
        'multi_in_strength': compute_multi_instrength(supra_adjacency, num_cells, num_layers),
        'multi_out_strength': compute_multi_outstrength(supra_adjacency, num_cells, num_layers),
        'multi_in_degree': compute_multi_indegree(supra_adjacency, num_cells, num_layers),
        'multi_out_degree': compute_multi_outdegree(supra_adjacency, num_cells, num_layers),
    }
    
    logger.info("✅ Network metrics computed")
    
    # Create enhanced analyzer
    analyzer = NetworkAnalyzer(
        experiment_name=args.experiment_name,
        logger=logger,
        default_figsize=(14, 10),  # Larger default figures
        default_dpi=300
    )
    
    # Custom plot configurations
    degree_config = {
        'figsize': (15, 8),
        'bins': 40,
        'alpha': 0.75,
        'violin_title': "Distribution of Multilayer Network Metrics",
        'xlabel': "Network Metric",
        'ylabel': "Metric Value",
        'title_fontsize': 16,
        'label_fontsize': 14
    }
    
    spatial_config = {
        'figsize': (24, 18),
        'scatter_size': 20,  # Smaller dots as requested
        'scatter_alpha': 0.65,
        'correlation_scatter_size': 15,  # Even smaller for correlations
        'xlabel': "Spatial X Coordinate (μm)",
        'ylabel': "Spatial Y Coordinate (μm)",
        'title_fontsize': 14,
        'label_fontsize': 12,
        'show_correlation_matrix': True
    }
    
    # Generate enhanced plots
    logger.info("Creating degree distribution plots")
    dist_plots = analyzer.plot_degree_distributions(metrics_data, degree_config)
    
    logger.info("Creating spatial network visualizations")
    
    # Create separate spatial plots for different metric groups
    basic_metrics = {k: v for k, v in metrics_data.items() 
                    if not k.startswith('multi_')}
    multi_metrics = {k: v for k, v in metrics_data.items() 
                    if k.startswith('multi_')}
    
    spatial_plots = []
    spatial_plots.extend(analyzer.plot_spatial_networks(
        subdata, cell_indexes, basic_metrics, spatial_config, "basic_metrics"
    ))
    spatial_plots.extend(analyzer.plot_spatial_networks(
        subdata, cell_indexes, multi_metrics, spatial_config, "multi_metrics"
    ))
    
    # Create correlation matrix
    logger.info("Creating correlation matrix")
    correlation_plots = analyzer.plot_correlation_matrix(
        metrics_data, 
        {'figsize': (12, 10), 'title_fontsize': 16}
    )
    
    logger.info(f"✅ Created {len(dist_plots)} distribution plots, "
               f"{len(spatial_plots)} spatial plots, and "
               f"{len(correlation_plots)} correlation plots")
    logger.info("✅ Enhanced network analysis and visualization completed")

