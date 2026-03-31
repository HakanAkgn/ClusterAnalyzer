import os
import numpy as np
from numba import njit

# Disable Numba cache by default
import os
if os.environ.get("NUMBA_CACHE") is None:
    os.environ["NUMBA_CACHE"] = 'False'

@njit(cache=eval(os.environ.get("NUMBA_CACHE")))
def find(parent, node):
    if parent[node] != node:
        parent[node] = find(parent, parent[node])
    return parent[node]

@njit(cache=eval(os.environ.get("NUMBA_CACHE")))
def union(parent, rank, size, node1, node2):
    root1, root2 = find(parent, node1), find(parent, node2)
    if root1 != root2:
        if rank[root1] > rank[root2]:
            parent[root2] = root1
            size[root1] += size[root2]
        elif rank[root1] < rank[root2]:
            parent[root1] = root2
            size[root2] += size[root1]
        else:
            parent[root2] = root1
            rank[root1] += 1
            size[root1] += size[root2]

@njit(cache=eval(os.environ.get("NUMBA_CACHE")))
def get_index(x, y, rows, cols):
    return (x % rows) * cols + (y % cols)

@njit(cache=eval(os.environ.get("NUMBA_CACHE")))
def union_neighbors(grid, rows, cols, parent, rank, size):
    for i in range(rows):
        for j in range(cols):
            right_neighbor = (i, (j + 1) % cols)
            bottom_neighbor = ((i + 1) % rows, j)

            if grid[i, j] == grid[right_neighbor]:
                union(parent, rank, size, get_index(i, j, rows, cols), get_index(*right_neighbor, rows, cols))
            if grid[i, j] == grid[bottom_neighbor]:
                union(parent, rank, size, get_index(i, j, rows, cols), get_index(*bottom_neighbor, rows, cols))

@njit(cache=eval(os.environ.get("NUMBA_CACHE")))
def calculate_cluster_sizes(grid, rows, cols, parent, size, state_cluster_counts):
    cluster_sizes = np.zeros(rows * cols, dtype=np.int32)

    for i in range(rows):
        for j in range(cols):
            root = find(parent, get_index(i, j, rows, cols))
            if cluster_sizes[root] == 0:
                cluster_sizes[root] = size[root]
                state = grid[i, j]
                state_cluster_counts[state] += 1

    return cluster_sizes

@njit(cache=eval(os.environ.get("NUMBA_CACHE")))
def calculate_size_counts(cluster_sizes):
    max_size = np.max(cluster_sizes)
    size_counts = np.zeros(max_size + 1, dtype=np.int32)

    for size_value in cluster_sizes:
        if size_value > 0:
            size_counts[size_value] += 1

    return size_counts




def analyze_clusters(grid, return_labeled_grid=False):

    def generate_labeled_grid(rows, cols, parent):
        # Generate a labeled grid for visualization
        labeled_grid = np.zeros(rows * cols, dtype=np.int32)
        label_map = {}
        current_label = 1
        for idx in range(rows * cols):
            root = find(parent, idx)
            if root not in label_map:
                label_map[root] = current_label
                current_label += 1
            labeled_grid[idx] = label_map[root]
        return labeled_grid.reshape((rows, cols))

    """
    Analyzes clusters in a grid using Union-Find algorithm.

    Parameters:
        grid (np.array): The grid containing different states.
        return_labeled_grid (bool): If True, returns the labeled grid for visualization.

    Returns:
        If return_labeled_grid is False:
            tuple: (size_counts_dict, state_cluster_counts_dict)
        If return_labeled_grid is True:
            np.array: labeled_grid
    """
    rows, cols = grid.shape
    parent = np.arange(rows * cols)
    rank = np.zeros(rows * cols, dtype=np.int32)
    size = np.ones(rows * cols, dtype=np.int32)
    state_cluster_counts = np.zeros(np.max(grid) + 1, dtype=np.int32)

    # Flatten the grid for easier indexing
    flat_grid = grid.flatten()

    # Perform union operations for neighboring cells with the same state
    union_neighbors(grid, rows, cols, parent, rank, size)

    if not return_labeled_grid:
        # Calculate cluster sizes and counts
        cluster_sizes = calculate_cluster_sizes(grid, rows, cols, parent, size, state_cluster_counts)
        size_counts = calculate_size_counts(cluster_sizes)

        size_counts_dict = {i: size_counts[i] for i in range(len(size_counts)) if size_counts[i] > 0}
        state_cluster_counts_dict = {i: state_cluster_counts[i] for i in range(len(state_cluster_counts)) if state_cluster_counts[i] > 0}

        return size_counts_dict, state_cluster_counts_dict
    else:
        # Generate labeled grid for visualization
        labeled_grid = generate_labeled_grid(rows, cols, parent)
        return labeled_grid

def box_counting(grid, min_box_size=1, max_box_size=None, step=1):
    """
    Performs box counting for fractal analysis on a grid.

    Parameters:
        grid (np.array): The grid to analyze.
        min_box_size (int): The minimum size of the box.
        max_box_size (int): The maximum size of the box.
        step (int): The step size to increase the box size.

    Returns:
        tuple: Two lists containing box sizes and corresponding counts.
    """
    nrows, ncols = grid.shape
    if max_box_size is None:
        max_box_size = min(nrows, ncols) // 2

    binary_grid = grid

    box_sizes = []
    box_counts = []

    box_size = min_box_size
    while box_size <= max_box_size:
        count = 0
        for i in range(0, nrows, box_size):
            for j in range(0, ncols, box_size):
                if np.any(binary_grid[i:i + box_size, j:j + box_size]):
                    count += 1
        box_sizes.append(box_size)
        box_counts.append(count)
        box_size += step

    return box_sizes, box_counts