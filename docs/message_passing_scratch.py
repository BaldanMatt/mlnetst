import marimo

__generated_with = "0.13.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import MessagePassing, global_mean_pool, GATConv
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.utils import to_undirected
    import torchvision.transforms as transforms
    from torchvision.datasets import MNIST
    import numpy as np
    from sklearn.neighbors import kneighbors_graph
    import matplotlib.pyplot as plt
    from pathlib import Path
    import os
    from torch.optim.lr_scheduler import StepLR
    import networkx as nx
    return (
        Data,
        DataLoader,
        F,
        GATConv,
        MNIST,
        MessagePassing,
        Path,
        StepLR,
        global_mean_pool,
        kneighbors_graph,
        mo,
        nn,
        np,
        nx,
        plt,
        to_undirected,
        torch,
        transforms,
    )


@app.cell
def _(F, GATConv, MessagePassing, att_weights, global_mean_pool, nn, torch):
    class AttentionMessagePassing(MessagePassing):
        def __init__(self, in_channels, out_channels, heads=1, dropout =0.0):
            super(AttentionMessagePassing, self).__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.heads = heads
            self.dropout = dropout

            # Linear transformation for queries, keys and values
            self.lin_q = nn.Linear(in_channels, out_channels*heads)
            self.lin_k = nn.Linear(in_channels, out_channels*heads)
            self.lin_v = nn.Linear(in_channels, out_channels*heads)

            # Output projection
            self.lin_out = nn.Linear(out_channels*heads, out_channels)

            # Attention parameters
            self.att_weight = nn.Parameter(torch.Tensor(1, heads, 2*out_channels))
            self.bias = nn.Parameter(torch.Tensor(out_channels))

            self.reset_parameters()

        def reset_parameters(self):
            nn.init.xavier_uniform_(self.lin_q.weight)
            nn.init.xavier_uniform_(self.lin_k.weight)
            nn.init.xavier_uniform_(self.lin_v.weight)
            nn.init.xavier_uniform_(self.lin_out.weight)
            nn.init.xavier_uniform_(self.att_weight)
            nn.init.zeros_(self.bias)

        def forward(self, x, edge_index):
            # Transform input features
            q = self.lin_q(x).view(-1, self.heads, self.out_channels)
            k = self.lin_k(x).view(-1, self.heads, self.out_channels)
            v = self.lin_v(x).view(-1, self.heads, self.out_channels)

            # Propagate messages
            out = self.propagate(edge_index, q=q, k=k, v=v)

            # Combine heads and apply output projection
            out = out.view(-1, self.heads*self.out_channels)
            out = self.lin_out(out) + self.bias

            return out

        def message(self, q_i, k_j, v_j, edge_index_i):
            # Compute attention weights
            att_input = torch.cat([q_i, k_j], dim=-1) # [E, heads, 2*out_channels]
            att_weigths = (att_input*self.att_weight).sum(dim=-1) # [E, heads]
            att_weights = F.softmax(att_weights, dim=-1)
            att_weights = F.dropout(att_weights, p=self.dropout, training=self.training)

            # apply attentino to values
            return att_weights.unsqueeze(-1)*v_j

    # Alternative using PyTorch Geometric's GATConv
    class GATLayer(nn.Module):
        def __init__(self, in_channels, out_channels, heads=1, dropout=0.0):
            super(GATLayer, self).__init__()
            self.gat = GATConv(in_channels, out_channels, heads=heads, dropout=dropout, concat=False)

        def forward(self, x, edge_index):
            return self.gat(x, edge_index)

    # Main GNN Model
    class MNISTGraphNet(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=64, num_classes=10, num_layers=3, heads=4, dropout=0.2):
            super(MNISTGraphNet, self).__init__()

            # Input projection
            self.input_proj = nn.Linear(input_dim, hidden_dim)

            # GNN layers with attention
            self.gnn_layers = nn.ModuleList()
            for i in range(num_layers):
                if i == 0:
                    # Use custom attention layer for first layer
                    self.gnn_layers.append(AttentionMessagePassing(hidden_dim, hidden_dim, heads, dropout))
                else:
                    # Use GATConv for subsequent layers
                    self.gnn_layers.append(GATLayer(hidden_dim, hidden_dim, heads, dropout))

            # Batch normalization and dropout
            self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
            self.dropout = nn.Dropout(dropout)

            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes)
            )

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch

            # Input projection
            x = self.input_proj(x)

            # Apply GNN layers
            for i, (gnn_layer, batch_norm) in enumerate(zip(self.gnn_layers, self.batch_norms)):
                residual = x
                x = gnn_layer(x, edge_index)
                x = batch_norm(x)
                x = F.relu(x)
                x = self.dropout(x)

                # Skip connection
                if i > 0:
                    x = x + residual

            # Global pooling
            x = global_mean_pool(x, batch)

            # Classification
            x = self.classifier(x)

            return x


    return (MNISTGraphNet,)


@app.cell
def _(mo):
    mo.md(
        r"""
    The GCN layer is mathematically defined as
    $x_i^{(k)} = \sum_{j\in\mathcal{N}(i)\cup{i}}\frac{1}{\sqrt{deg(i)}\cdot\sqrt{deg(j)}}\cdot(\mathbf{W}^T\cdot(x_j^{(k-1)}))+b$ where neighbouring features are first transformed by a weight matrix W, normalized by their degree, and finally summed up.

    Lastly, we apply the bias vector b to the aggregated output.

    - Add self loops to the adjacency matrix
    - Linearly trnasform node feature matrix
    - Compute normalization coefficeints
    - Normalize features in $\psi$
    - Sum up neighboring node features
    - Apply a final bias vector

    Graph convolution is analogous to consider convolution on images where pixels are adjacent to other pixels. Convolution is a simple sliding window over the whole image that multiplies the image pixels with the filter weights. Similarly, graph convolution uses information from the neighboring nodes to predict features of a given node $x_i$.

    A function is needed to transform node features into a latent space $h_i$, which can then be used for further algorithmic computation.

    GNN architectures can mainly be categorized into Spectral, Spatial and Sampling methods.
    [[https://www.v7labs.com/blog/graph-neural-networks-guide]]

    Convolution is performed for eahc node in the convolution layer graph. The feature information from the neighbors of the nodes is aggregated and the nodes ar eupdated likewise.
    Next, a non-linear activation function such as ReLU is applied to the ouput of the convolution layer.

    GCN do not support edge features and the notion of message passing is non-existent.

    Tasks may range between:
    - Link Prediction
    - Node Classification
    - Clustering
    - Graph classification
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Deep dive into Message Passing

    **Message passing** is the most important component in GNN. It is actually a mathematical function f() that updates the receiver node by using the messages from each neighboring sender node.

    Understand the message passing function as an "Aggregation Function"

    $f(x_1) = \sigma(c_{1,1}x_1W+c_{1,2}x_2W + c_{1,3}x_3W + c_{1,4}x_4W)$

    Where 2<->1, 3<->1, 4<->1 are the edges of the graph.
    c_{i,j} are constant scalar values.
    W are the weights matrices that the GNN can learn through backpropagation to know which features are more important.

    Non-linearity is introduced by the sigma function.

    We MUST require a **permutation-invariant function that consistently aggregates information** from a node's neighborhood regardless of the ordering.

    $f(x_i) = \phi(x_i, g(c_{ij}\psi(x_j))$ where $g$ is an aggregation of all the neighbours of i. $\psi$ is a function that denotes the weights.

    $\phi$ can be a general learnable function.

    So, in the above message passing function weights of the neighbors are fixed based on the structure of the graph which is one of the three general flavors of graph neural networks layers called Convolutional.

    ## Attentional GNN

    We can also learn the weights to each neighbor accoridng to their features, which is another flavor of Graph Neural Networks layers called Attentional.

    In these layers, weights of neighbors are learned based on the interactions of the features between the nodes.
    $$f(x_i) = \phi(x_i, g(a(x_i, x_j)\psi(x_j))$$

    Which in the most gneeral way is incorporated in $\psi$ as $$f(x_i) = \phi(x_i, g(\psi(x_i,x_j))$$

    Computing the updates for each node sequentially is a slow process, so in practive we use linear algebra to fasten this whole process of message passing. We summarize the grpahs edges using a table we call an Adjacency matrix.

    We also have two more matrices that are known as Fetures Matrix and Weights Matrix. Feature matrix contains all the features of nodes while the weight matrix contains the learnable weights.

    ![Matrices](https://miro.medium.com/v2/resize:fit:960/format:webp/1*M2wkBndEj3CqItscVWjTSw.png)
    ![Generalized Matrices](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*V-TcO4kUfFIvykkRQx7m6w.png)

    With this, we can use a matrix multiplication library to perform message passing for all the nodes at once as $$f(X) = \sigma(AXW)$$
    """
    )
    return


@app.cell
def _(
    Data,
    MNIST,
    Path,
    StepLR,
    kneighbors_graph,
    nn,
    np,
    nx,
    plt,
    to_undirected,
    torch,
    transforms,
):
    # Data preprocessing functions
    def create_graph_from_image(image, k=8):
        """Convert MNIST image to graph structure"""
        # Flatten image and get coordinates
        h, w = image.shape
        coords = np.array([[i, j] for i in range(h) for j in range(w)])

        # Create k-nearest neighbors graph based on spatial coordinates
        knn_graph = kneighbors_graph(coords, n_neighbors=k, mode='connectivity', include_self=False)
        edge_index = torch.tensor(np.array(knn_graph.nonzero()), dtype=torch.long)

        # Make graph undirected
        edge_index = to_undirected(edge_index)

        # Node features (pixel values)
        node_features = torch.tensor(image.flatten(), dtype=torch.float).unsqueeze(1)

        return edge_index, node_features

    def mnist_to_graph_data(dataset, k=8, threshold=0.1):
        """
        Convert MNIST dataset to graph format with proper edge index validation.
    
        Args:
            dataset: MNIST dataset
            k: number of nearest neighbors for graph construction
            threshold: minimum pixel value to consider as a node
        """
        graph_data_list = []
    
        for i, (image, label) in enumerate(dataset):
            if i % 100 == 0:
                print(f"Processing image {i}/{len(dataset)}")
        
            # Convert image to numpy array
            if isinstance(image, torch.Tensor):
                image = image.numpy()
        
            # Flatten image and get coordinates
            if len(image.shape) == 3:
                image = image.squeeze()
        
            # Create node features from non-zero pixels
            node_coords = []
            node_features = []
        
            for row in range(image.shape[0]):
                for col in range(image.shape[1]):
                    if image[row, col] > threshold:
                        node_coords.append([row, col])
                        node_features.append([image[row, col]])
        
            # Skip if too few nodes
            if len(node_coords) < 2:
                print(f"Warning: Image {i} has only {len(node_coords)} nodes, skipping")
                continue
        
            node_coords = np.array(node_coords)
            node_features = np.array(node_features)
        
            # Create k-NN graph
            try:
                # Adjust k if we have fewer nodes than k
                actual_k = min(k, len(node_coords) - 1)
                knn_graph = kneighbors_graph(node_coords, n_neighbors=actual_k, 
                                           mode='connectivity', include_self=False)
            
                # Convert to edge index format
                edge_index = np.array(knn_graph.nonzero())
            
                # Validate edge indices
                max_node_idx = len(node_coords) - 1
                if edge_index.max() > max_node_idx:
                    print(f"Error: Edge index {edge_index.max()} > max node index {max_node_idx}")
                    continue
            
                # Create PyTorch Geometric Data object
                graph_data = Data(
                    x=torch.tensor(node_features, dtype=torch.float),
                    edge_index=torch.tensor(edge_index, dtype=torch.long),
                    y=torch.tensor(label, dtype=torch.long)
                )
            
                # Additional validation
                num_nodes = graph_data.x.size(0)
                max_edge_idx = graph_data.edge_index.max().item()
            
                if max_edge_idx >= num_nodes:
                    print(f"Validation failed: max edge index {max_edge_idx} >= num nodes {num_nodes}")
                    continue
            
                graph_data_list.append(graph_data)
            
            except Exception as e:
                print(f"Error processing image {i}: {e}")
                continue
    
        print(f"Successfully converted {len(graph_data_list)} images to graphs")
        return graph_data_list

    def demonstrate_graph_construction():
        """Demonstrate the graph construction process with debug information"""
        print("=== Graph Construction Demonstration ===")
    
        # Load a single MNIST image for demonstration
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.squeeze())])
        dataset = MNIST(root="./data", train=True, download=True, transform=transform)
    
        # Take first image
        image, label = dataset[0]
        print(f"Original image shape: {image.shape}")
        print(f"Image label: {label}")
        print(f"Image min/max values: {image.min():.3f}/{image.max():.3f}")
    
        # Convert to graph
        single_dataset = [(image, label)]
        graph_data = mnist_to_graph_data(single_dataset, k=8, threshold=0.1)
    
        if graph_data:
            graph = graph_data[0]
            print(f"\nGraph properties:")
            print(f"Number of nodes: {graph.x.size(0)}")
            print(f"Number of edges: {graph.edge_index.size(1)}")
            print(f"Node features shape: {graph.x.shape}")
            print(f"Edge index shape: {graph.edge_index.shape}")
            print(f"Edge index min/max: {graph.edge_index.min()}/{graph.edge_index.max()}")
            print(f"Label: {graph.y}")
        
            # Visualize the graph
            plt.figure(figsize=(15, 5))
        
            # Original image
            plt.subplot(1, 3, 1)
            plt.imshow(image, cmap='gray')
            plt.title(f'Original Image (Label: {label})')
            plt.axis('off')
        
            # Nodes visualization
            plt.subplot(1, 3, 2)
            plt.imshow(image, cmap='gray', alpha=0.3)
        
            # Extract node coordinates from the graph construction
            node_coords = []
            for row in range(image.shape[0]):
                for col in range(image.shape[1]):
                    if image[row, col] > 0.1:
                        node_coords.append([row, col])
        
            if node_coords:
                node_coords = np.array(node_coords)
                plt.scatter(node_coords[:, 1], node_coords[:, 0], c='red', s=2, alpha=0.7)
        
            plt.title(f'Graph Nodes ({len(node_coords)} nodes)')
            plt.axis('off')
        
            # Edge visualization (simplified)
            plt.subplot(1, 3, 3)
            plt.imshow(image, cmap='gray', alpha=0.3)
        
            # Draw some edges (sample to avoid overcrowding)
            edge_index = graph.edge_index.numpy()
            sample_edges = np.random.choice(edge_index.shape[1], 
                                          min(100, edge_index.shape[1]), 
                                          replace=False)
        
            for edge_idx in sample_edges:
                src, dst = edge_index[:, edge_idx]
                if src < len(node_coords) and dst < len(node_coords):
                    plt.plot([node_coords[src, 1], node_coords[dst, 1]], 
                            [node_coords[src, 0], node_coords[dst, 0]], 
                            'b-', alpha=0.3, linewidth=0.5)
        
            plt.scatter(node_coords[:, 1], node_coords[:, 0], c='red', s=2, alpha=0.7)
            plt.title(f'Graph Edges (sample of {len(sample_edges)} edges)')
            plt.axis('off')
        
            plt.tight_layout()
            plt.show()
        
            return graph_data
        else:
            print("Failed to create graph from image")
            return None

    def validate_graph_data(graph_data_list):
        """Validate all graphs in the dataset"""
        print("=== Validating Graph Dataset ===")
    
        valid_graphs = []
        for i, graph in enumerate(graph_data_list):
            num_nodes = graph.x.size(0)
            max_edge_idx = graph.edge_index.max().item() if graph.edge_index.size(1) > 0 else -1
        
            if max_edge_idx >= num_nodes:
                print(f"Invalid graph {i}: max edge index {max_edge_idx} >= num nodes {num_nodes}")
            else:
                valid_graphs.append(graph)
    
        print(f"Valid graphs: {len(valid_graphs)}/{len(graph_data_list)}")
        return valid_graphs
            
            # Create graph
            #edge_index, node_features = create_graph_from_image(image_array, k)

            # Create Data object
            #data = Data(x=node_features, edge_index=edge_index, y=torch.tensor(label, dtype=torch.long))
            #graph_data_list.append(data)

        #return graph_data_list

    def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001, device='cuda'):
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
        criterion = nn.CrossEntropyLoss()

        train_losses = []
        val_accuracies = []

        for epoch in range(num_epochs):
            # Training
            model.train()
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()

                output = model(batch)
                loss = criterion(output, batch.y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)

            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    output = model(batch)
                    _, predicted = torch.max(output.data, 1)
                    total += batch.y.size(0)
                    correct += (predicted == batch.y).sum().item()

            val_accuracy = 100 * correct / total
            val_accuracies.append(val_accuracy)

            scheduler.step()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

        return train_losses, val_accuracies

    def visualize_mnist_graph(image, label, k=8, threshold=0.1):
        """Visualize MNIST image and its graph representation"""
        # Convert to graph
        edge_index, node_features = create_graph_from_image(image, k)

        # Get coordinates for visualization
        h, w = image.shape
        coords = np.array([[i, j] for i in range(h) for j in range(w)])

        # Filter nodes with intensity above threshold for cleaner visualization
        node_intensities = node_features.squeeze().numpy()
        significant_nodes = node_intensities > threshold

        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1. Original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title(f'Original MNIST Image\nLabel: {label}')
        axes[0].axis('off')

        # 2. Graph nodes (all pixels)
        scatter = axes[1].scatter(coords[:, 1], coords[:, 0], c=node_intensities, 
                                cmap='gray', s=15, alpha=0.7)
        axes[1].set_title(f'All Graph Nodes\nNodes: {len(coords)}')
        axes[1].set_xlabel('x coordinate')
        axes[1].set_ylabel('y coordinate')
        axes[1].invert_yaxis()  # Match image orientation
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1], label='Pixel intensity')

        # 3. Graph with edges (filtered for significant nodes)
        if len(edge_index[0]) > 0:
            # Create NetworkX graph for visualization
            G = nx.Graph()

            # Add all nodes with positions
            pos = {}
            for i, (y, x) in enumerate(coords):
                G.add_node(i, intensity=node_intensities[i])
                pos[i] = (x, -y)  # Flip y to match image orientation

            # Add edges
            edges = edge_index.numpy()
            edge_list = []
            for i in range(edges.shape[1]):
                src, dst = edges[0, i], edges[1, i]
                # Only add edges between significant nodes for cleaner visualization
                if significant_nodes[src] and significant_nodes[dst]:
                    G.add_edge(src, dst)
                    edge_list.append((src, dst))

            # Filter to only show significant nodes
            significant_node_ids = [i for i in range(len(coords)) if significant_nodes[i]]
            G_filtered = G.subgraph(significant_node_ids)
            pos_filtered = {i: pos[i] for i in significant_node_ids}

            # Draw graph
            node_colors = [node_intensities[i] for i in G_filtered.nodes()]
            nx.draw(G_filtered, pos_filtered, ax=axes[2], node_color=node_colors, 
                    cmap='gray', node_size=20, width=0.3, alpha=0.8, edge_color='red')
            axes[2].set_title(f'Graph Structure (Significant Nodes)\nNodes: {len(G_filtered.nodes())}, Edges: {len(G_filtered.edges())}')
            axes[2].set_aspect('equal')
        else:
            axes[2].text(0.5, 0.5, 'No edges found', 
                        ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('Graph Structure')

        plt.tight_layout()
        plt.show()

        # Print graph statistics
        print(f"\nGraph Statistics:")
        print(f"Label: {label}")
        print(f"Total nodes: {len(coords)}")
        print(f"Significant nodes (>{threshold}): {significant_nodes.sum()}")
        print(f"Total edges: {edge_index.shape[1]}")
        print(f"Node feature dimensions: {node_features.shape[1]}")
        print(f"Average node degree: {2 * edge_index.shape[1] / len(coords):.2f}")

        return edge_index, node_features

    def visualize_attention_weights(model, data, layer_idx=0, head_idx=0):
        """Visualize attention weights for a specific layer and head"""
        model.eval()

        # Forward pass with hooks to capture attention weights
        attention_weights = {}

        def attention_hook(module, input, output):
            if hasattr(module, 'att_weight'):
                # For custom attention layer
                attention_weights['custom'] = module.last_attention_weights

        # Register hooks
        if layer_idx < len(model.gnn_layers):
            hook = model.gnn_layers[layer_idx].register_forward_hook(attention_hook)

        # Forward pass
        with torch.no_grad():
            output = model(data)

        # Remove hooks
        if 'hook' in locals():
            hook.remove()

        print(f"Model output shape: {output.shape}")
        print(f"Predicted class: {torch.argmax(output, dim=1).item()}")

        return output

    def demonstrate_visualization():
        """Demonstrate the visualization functions"""
        # Load MNIST dataset
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.squeeze())])
        test_dataset = MNIST(root=str(Path(__file__).parents[1]/"data"/"raw"), train=False, download=True, transform=transform)

        # Get a few sample images
        sample_indices = [0, 1]  # First 5 test images

        for idx in sample_indices:
            image, label = test_dataset[idx]
            image_np = np.array(image)

            print(f"\n{'='*50}")
            print(f"Visualizing sample {idx}")
            visualize_mnist_graph(image_np, label, k=8, threshold=0.1)

            # Break after first sample to avoid too much output
            if idx == 0:
                break
    return (
        demonstrate_visualization,
        mnist_to_graph_data,
        train_model,
        validate_graph_data,
    )


@app.cell
def _(
    Data,
    DataLoader,
    MNIST,
    MNISTGraphNet,
    Path,
    demonstrate_visualization,
    mnist_to_graph_data,
    plt,
    torch,
    train_model,
    transforms,
    validate_graph_data,
):
    def main(train_subset_size,test_subset_size):
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Demonstrate visualization first
        print("Demonstrating graph visualization...")
        demonstrate_visualization()

        # Load MNIST dataset
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.squeeze())])

        # Use smaller subset for demonstration
        train_dataset = MNIST(root=str(Path(__file__).parents[1] / "data" / "raw"), train=True, download=True, transform=transform)
        test_dataset = MNIST(root=str(Path(__file__).parents[1] / "data" / "raw"), train=False, download=True, transform=transform)

        # Create smaller datasets for faster processing
        train_subset = torch.utils.data.Subset(train_dataset, range(train_subset_size))  # Use first 1000 samples
        test_subset = torch.utils.data.Subset(test_dataset, range(test_subset_size))     # Use first 200 samples

        print("\nConverting MNIST to graph format...")
        train_graph_data = mnist_to_graph_data(train_subset, k=8, threshold=0.1)
        test_graph_data = mnist_to_graph_data(test_subset, k=8, threshold = 0.1)

        train_graph_data = validate_graph_data(train_graph_data)
        test_graph_data = validate_graph_data(test_graph_data)

        # Create data loaders
        train_loader = DataLoader(train_graph_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_graph_data, batch_size=32, shuffle=False)

        # Create model
        model = MNISTGraphNet(
            input_dim=1,
            hidden_dim=64,
            num_classes=10,
            num_layers=3,
            heads=4,
            dropout=0.2
        )

        print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

        # Train model
        print("Starting training...")
        train_losses, val_accuracies = train_model(
            model, train_loader, test_loader, 
            num_epochs=20, lr=0.001, device=device
        )

        # Plot results
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies)
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')

        plt.tight_layout()
        plt.show()

        # Final evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                output = model(batch)
                _, predicted = torch.max(output.data, 1)
                total += batch.y.size(0)
                correct += (predicted == batch.y).sum().item()

        final_accuracy = 100 * correct / total
        print(f'Final Test Accuracy: {final_accuracy:.2f}%')

        # Demonstrate model prediction on a sample
        print("\nDemonstrating model prediction...")
        sample_data = test_graph_data[0]
        sample_batch = Data(x=sample_data.x, edge_index=sample_data.edge_index, 
                           batch=torch.zeros(sample_data.x.size(0), dtype=torch.long))

        model.eval()
        with torch.no_grad():
            output = model(sample_batch.to(device))
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1).max().item()

        print(f"True label: {sample_data.y.item()}")
        print(f"Predicted label: {predicted_class}")
        print(f"Confidence: {confidence:.3f}")

    return (main,)


@app.cell
def _(main):
    main(
        train_subset_size=50,
        test_subset_size=20
    )
    return


if __name__ == "__main__":
    app.run()
