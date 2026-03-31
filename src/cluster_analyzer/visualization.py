import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from .cluster import analyze_clusters  # Adjust the import based on your package structure
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

def plot_clusters(grid, n_clusters=5, cmap=None, save_path=None, figsize=(8, 16)):
    """
    Plots the real space grid and the first n largest clusters.

    Parameters:
        grid (np.array): The input grid representing the state matrix.
        n_clusters (int): Number of top clusters to plot.
        cmap (ListedColormap): Custom colormap for plotting clusters.
        save_path (str): File path to save the plot. If None, the plot is not saved.
        figsize (tuple): Size of the figure (width, height) in inches.

    Returns:
        None: Displays the plot and optionally saves it to a file.
    """
    if cmap is None:
        # Base colors for clusters
        base_colors = [
            [1, 1, 1],          # White for background
            [0.698, 0.133, 0.133],  # Firebrick
            [0.6, 0.196, 0.8],      # Purple
            [0.275, 0.51, 0.706],   # Steel Blue
            [0.196, 0.804, 0.196],  # Lime Green
            [1, 0.843, 0],          # Gold
            [0.545, 0.271, 0.075],  # Sienna
            [0.25, 0.878, 0.815],   # Turquoise
            [0.941, 0.502, 0.502],  # Light Coral
            [0.5, 0.5, 0.5]         # Gray
        ]
        # Extend base colors if needed
        if n_clusters > len(base_colors) - 1:
            # Repeat colors if not enough unique colors are available
            base_colors = base_colors + base_colors[1:] * ((n_clusters // len(base_colors)) + 1)
        cmap = ListedColormap(base_colors[:n_clusters + 1])

    # Get the labeled grid from analyze_clusters
    labeled_grid = analyze_clusters(grid, return_labeled_grid=True)

    # Get unique labels and their counts
    unique_labels, counts = np.unique(labeled_grid, return_counts=True)

    # Exclude background label (0)
    mask = unique_labels != 0
    unique_labels = unique_labels[mask]
    counts = counts[mask]

    # Sort clusters by size (descending order)
    sorted_indices = np.argsort(counts)[::-1]
    top_clusters_labels = unique_labels[sorted_indices][:n_clusters]

    # Create a cluster image highlighting the top n clusters
    cluster_image = np.zeros_like(grid)
    for idx, cluster_label in enumerate(top_clusters_labels, start=1):
        cluster_image[labeled_grid == cluster_label] = idx

    # Create the plot
    fig, axes = plt.subplots(2, 1, figsize=figsize, constrained_layout=True)

    # Plot the state matrix
    ax0 = axes[0]
    im0 = ax0.imshow(grid, cmap='viridis')
    ax0.set_title('State Matrix')
    ax0.axis('equal')
    ax0.axis('off')
    ax0.add_patch(Rectangle(
        (0, 0),
        grid.shape[1],
        grid.shape[0],
        fill=False,
        edgecolor='black',
        lw=2
    ))
    plt.colorbar(im0, ax=ax0)

    # Plot the cluster image
    ax1 = axes[1]
    im1 = ax1.imshow(cluster_image, cmap=cmap, vmin=0, vmax=n_clusters)
    ax1.set_title(f'Top {n_clusters} Clusters')
    ax1.axis('equal')
    ax1.axis('off')
    ax1.add_patch(Rectangle(
        (0, 0),
        cluster_image.shape[1],
        cluster_image.shape[0],
        fill=False,
        edgecolor='black',
        lw=2
    ))
    plt.colorbar(im1, ax=ax1, ticks=range(n_clusters + 1))

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()