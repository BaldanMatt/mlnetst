import marimo

__generated_with = "0.13.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Data, DataLoader
    from torchvision import datasets, transforms
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    from sklearn.neighbors import kneighbors_graph
    from pathlib import Path
    from torch_geometric.utils import to_undirected
    import networkx as nx
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import umap
    return (
        Data,
        DataLoader,
        F,
        GCNConv,
        NearestNeighbors,
        Path,
        datasets,
        global_mean_pool,
        kneighbors_graph,
        nn,
        np,
        nx,
        plt,
        to_undirected,
        torch,
        tqdm,
        transforms,
        umap,
    )


@app.cell
def _(
    Data,
    NearestNeighbors,
    kneighbors_graph,
    np,
    nx,
    plt,
    to_undirected,
    torch,
):
    def image_to_graph(image, k=8):
        """
        Convert MNIST image to graph representation
    
        Args:
            image: 28x28 numpy array
            k: number of nearest neighbors for each pixel
    
        Returns:
            torch_geometric.data.Data object
        """
        h, w = image.shape
    
        # Create node features (pixel intensities)
        node_features = image.flatten().reshape(-1, 1)  # Shape: (784, 1)
    
        # Create coordinate matrix for each pixel
        coords = []
        for i in range(h):
            for j in range(w):
                coords.append([i, j])
        coords = np.array(coords)
    
        # Find k-nearest neighbors for each pixel based on spatial distance
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords)
        distances, indices = nbrs.kneighbors(coords)
    
        # Create edge list (exclude self-connections)
        edge_list = []
        for i in range(len(indices)):
            for j in range(1, len(indices[i])):  # Skip first index (self)
                edge_list.append([i, indices[i][j]])
    
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        node_features = torch.tensor(node_features, dtype=torch.float)
    
        return Data(x=node_features, edge_index=edge_index)

    def create_graph_dataset(dataset, max_samples=1000):
        """
        Convert MNIST dataset to graph dataset
        """
        graph_data = []
        labels = []
    
        for i, (image, label) in enumerate(dataset):
            if i >= max_samples:
                break
            
            # Convert PIL image to numpy array
            image_np = image.numpy().squeeze()
        
            # Create graph from image
            graph = image_to_graph(image_np)
            if i == 0:
                visualize_mnist_graph(image_np, label)
            graph.y = torch.tensor([label], dtype=torch.long)
        
            graph_data.append(graph)
            labels.append(label)

            if max_samples > 100:
                if i % 100 == 0:
                    print(f"Processed {i+1}/{max_samples} samples")
            else:
                if i % 10 == 0:
                    print(f"Processed {i+1}/{max_samples} samples")
    
        return graph_data

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

    return (create_graph_dataset,)


@app.cell
def _(np, plt, torch, tqdm, umap):
    def extract_embeddings(model, loader, device):
        """Extract embeddings and labels from the model"""
        model.eval()
        embeddings = []
        labels = []
    
        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting embeddings"):
                batch = batch.to(device)
                _, batch_embeddings = model(batch.x, batch.edge_index, batch.batch, return_embeddings=True)
            
                embeddings.append(batch_embeddings.cpu().numpy())
                labels.append(batch.y.cpu().numpy())
    
        embeddings = np.vstack(embeddings)
        labels = np.concatenate(labels)
    
        return embeddings, labels

    def visualize_embeddings(embeddings, labels, title="GNN Embeddings", save_path=None):
        """Create UMAP visualization of embeddings colored by labels"""
        print("Computing UMAP projection...")
    
        # Fit UMAP
        umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embeddings_2d = umap_model.fit_transform(embeddings)
    
        # Create the plot
        plt.figure(figsize=(12, 8))
    
        # Create scatter plot with different colors for each digit
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
        for digit in range(10):
            mask = labels == digit
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=[colors[digit]], label=f'Digit {digit}', 
                       alpha=0.7, s=20)
    
        plt.title(title, fontsize=16)
        plt.xlabel('UMAP 1', fontsize=12)
        plt.ylabel('UMAP 2', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
        return embeddings_2d
    return extract_embeddings, visualize_embeddings


@app.cell
def _(F, GCNConv, global_mean_pool, nn, torch, tqdm):
    class SimpleGNN(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=64, output_dim=10):
            super(SimpleGNN, self).__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.classifier = nn.Linear(hidden_dim, output_dim)

        def forward(self, x, edge_index, batch, return_embeddings=False):
            # apply layers
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))

            # Global pooling to get graph-level representation
            embeddings = global_mean_pool(x, batch)
        
            # Classification
            logits = self.classifier(embeddings)
            output = F.log_softmax(logits, dim=1)
        
            if return_embeddings:
                return output, embeddings
            return output

    def train_epoch(model, loader, optimizer, device):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        pbar = tqdm(loader, desc="Training", leave=False)
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
        
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.nll_loss(out, batch.y)
        
            loss.backward()
            optimizer.step()
        
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
        
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })
    
        return total_loss / len(loader), correct / total

    def test_epoch(model, loader, device):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = F.nll_loss(out, batch.y)

                total_loss += loss.item()
                pred = out.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)

        return total_loss / len(loader), correct / total
            
    return SimpleGNN, test_epoch, train_epoch


@app.cell
def _(
    DataLoader,
    Path,
    SimpleGNN,
    create_graph_dataset,
    datasets,
    extract_embeddings,
    test_epoch,
    torch,
    train_epoch,
    transforms,
    visualize_embeddings,
):
    def main():
            # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
    
        # Load MNIST dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
        train_dataset = datasets.MNIST(str(Path(__file__).parents[1] / "data" / "raw"), train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(str(Path(__file__).parents[1] / "data" / "raw"), train=False, transform=transform)
    
        # Convert to graph datasets (using smaller subsets for demonstration)
        print("Converting training images to graphs...")
        train_graphs = create_graph_dataset(train_dataset, max_samples=1000)
    
        print("Converting test images to graphs...")
        test_graphs = create_graph_dataset(test_dataset, max_samples=200)
    
        train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

        # Init model
        model = SimpleGNN(input_dim = 1, hidden_dim = 64, output_dim = 10).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training loop
        num_epochs = 10
        print(f"Starting training for {num_epochs} epochs...")
    
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
            test_loss, test_acc = test_epoch(model, test_loader, device)
        
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
        print("\nTraining completed!")
    
        # Visualize embeddings
        print("\nExtracting and visualizing embeddings...")
    
        # Extract embeddings from training data
        train_embeddings, train_labels = extract_embeddings(model, train_loader, device)
    
        # Create UMAP visualization
        train_embeddings_2d = visualize_embeddings(
            train_embeddings, train_labels, 
            title="GNN Training Embeddings (UMAP Projection)",
            save_path="gnn_train_embeddings.png"
        )
    
        # Also visualize test embeddings
        test_embeddings, test_labels = extract_embeddings(model, test_loader, device)
        test_embeddings_2d = visualize_embeddings(
            test_embeddings, test_labels,
            title="GNN Test Embeddings (UMAP Projection)",
            save_path="gnn_test_embeddings.png"
        )
    
        # Print some statistics
        print(f"\nEmbedding Statistics:")
        print(f"Training embeddings shape: {train_embeddings.shape}")
        print(f"Test embeddings shape: {test_embeddings.shape}")
        print(f"Embedding dimension: {train_embeddings.shape[1]}")
    
        return model, train_embeddings_2d, test_embeddings_2d
    
    return (main,)


@app.cell
def _(main):
    main()
    return


if __name__ == "__main__":
    app.run()
