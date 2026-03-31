import numpy as np
from scipy.signal import convolve2d
import sympy as sp
from .cluster import analyze_clusters, box_counting
from .activity import calculate_activity_rate
import csv
from tqdm import trange, tqdm

def GOL(grid, t):
    """
    Classic Game of Life function.

    Parameters:
        grid (np.array): The initial grid of cells, with 1 representing alive and 0 dead cells.
        t (int): The number of time steps to evolve the game.

    Returns:
        np.array: The final grid after t time steps.
    """
    # Define the 3x3 kernel to count neighbors (Moore Neighborhood)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    for _ in range(t):
        # Step 1: Convolve the grid with the kernel to count alive neighbors for each cell
        b = convolve2d(grid, kernel, mode='same', boundary='wrap')
        # Step 2: Apply Game of Life rules
        # Birth Rule: A dead cell with exactly 3 neighbors becomes alive
        # Survival Rule: An alive cell with 2 or 3 neighbors stays alive
        # Death Rule: All other cells die
        grid = ((grid == 1) & ((b == 2) | (b == 3))) | ((grid == 0) & (b == 3))

        # Convert boolean array to integer array (1 for alive, 0 for dead)
        grid = grid.astype(int)
    return grid




def generate_cantor_set(order, lamda):
    v = np.array([0, 1])
    for _ in range(order):
        v = np.concatenate(((1 - lamda) * v, v + lamda * (1 - v)))
    return v

def LogisticGOL(grid, t, t1, t2, t3, order, lamda):
    """
    Logistic Game of Life (LogisticGOL) function implementing custom growth, stability, and decay rules.

    Parameters:
        grid (np.array): Initial grid of states, where each cell has an integer state.
        t (int): Number of time steps to iterate the grid.
        t1 (float): Lower threshold for stability.
        t2 (float): Lower threshold for growth.
        t3 (float): Upper threshold for growth.
        order (int): Order of the Cantor set generation, controlling the resolution.
        lamda (float): Scaling factor in Cantor set generation, affecting the recursive pattern.

    Returns:
        np.array: The final grid after t time steps.

    Steps:
    1. **Cantor Set Generation:** Using the given `order` and `lamda` values, a recursive process generates a scaled Cantor set for the `grid`.
    2. **Grid Update Loop:** The grid evolves over `t` time steps according to logistic rules.
    """

 
    # Define the 3x3 kernel to count neighbors (Moore Neighborhood)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    L = 2**(order + 1)  # Cantor set size for the given order
    cantor = generate_cantor_set(order, lamda)

    for _ in range(t):
        # Map grid states to values in the Cantor set
        s = cantor[grid]
        # Apply convolution to get the sum of the Moore neighborhood
        b = convolve2d(s, kernel, mode='same', boundary='wrap')
        # Update grid based on growth, stability, and decay rules
        grid = (L // 2 + grid // 2) * (b >= t2) * (b <= t3) + (grid) * (b >= t1) * (b < t2) + (grid // 2) * ((b < t1) + (b > t3))
    return grid






def lambda_neighborhood_transitions(lambda_value, target_value, num_states, max_order, tolerance):
    """
    Generates Cantor sets and finds a combination of states that sum up to a target value
    within a given tolerance, using both numerical and symbolic computations.

    Parameters:
        lambda_value (float): The λ value used in Cantor set generation.
        target_value (float): The target sum value to achieve with the selected states.
        num_states (int): The number of states to select.
        max_order (int): Maximum order for Cantor set generation.
        tolerance (float): Allowed deviation from the target value.

    Returns:
        dict: A dictionary containing selected numerical states, symbolic expressions,
              total numerical sum, simplified symbolic expression, and its evaluated value.
    """

    def generate_cantor_set(order, lamda):
        v = np.array([0.0, 1.0])
        for _ in range(order):
            v = np.concatenate(((1 - lamda) * v, v + lamda * (1 - v)))
        return v

    def generate_cantor_set_symbolic(order):
        λ = sp.Symbol('λ')
        v = np.array([0, 1], dtype=object)
        for _ in range(order):
            v = np.concatenate(((1 - λ) * v, v + λ * (1 - v)))
        return v

    def generate_cantor_states_with_symbolic(max_order, lambda_value):
        symbolic_states = []
        numerical_states = []
        lamda = float(lambda_value)
        λ = sp.Symbol('λ')

        for n in range(1, max_order + 1):
            # Generate numerical states
            v_num = generate_cantor_set(n, lamda)
            numerical_states.extend(v_num)
            # Generate symbolic states
            v_sym = generate_cantor_set_symbolic(n)
            symbolic_states.extend(v_sym)

        # Remove duplicates and sort
        numerical_states, indices = np.unique(numerical_states, return_index=True)
        symbolic_states = np.array(symbolic_states)[indices]

        # Create a mapping from numerical to symbolic states
        state_mapping = dict(zip(numerical_states, symbolic_states))

        return numerical_states, symbolic_states, state_mapping

    def subset_sum_branch_and_bound(states, target_value, num_states, tolerance):
        states = sorted(states)
        n = len(states)
        result = []
        min_deviation = float('inf')

        def dfs(start, path, total):
            nonlocal min_deviation, result

            # Prune branches that are too long or too short
            if len(path) > num_states or total - target_value > tolerance:
                return
            if len(path) == num_states:
                deviation = abs(total - target_value)
                if deviation <= tolerance and deviation < min_deviation:
                    min_deviation = deviation
                    result = list(path)
                return
            for i in range(start, n):
                # Prune if adding the smallest possible sum exceeds target + tolerance
                if total + (num_states - len(path)) * states[i] - target_value > tolerance:
                    break
                # Prune if adding the largest possible sum is less than target - tolerance
                max_possible = total + sum(states[-(num_states - len(path)):])
                if max_possible - target_value < -tolerance:
                    return
                dfs(i, path + [states[i]], total + states[i])

        dfs(0, [], 0)
        return result

    # Generate numerical and symbolic states
    numerical_states, symbolic_states, state_mapping = generate_cantor_states_with_symbolic(max_order, lambda_value)

    # Use the numerical states for the subset sum algorithm
    selected_states = subset_sum_branch_and_bound(numerical_states, target_value, num_states, tolerance)

    if selected_states:
        total_value = sum(selected_states)
        symbolic_selected_states = [state_mapping[num_state] for num_state in selected_states]

        # Sum the symbolic expressions
        total_symbolic_expr = sum(symbolic_selected_states)
        simplified_expr = sp.simplify(total_symbolic_expr)

        # Evaluate the simplified expression numerically to verify
        simplified_value = simplified_expr.evalf(subs={'λ': lambda_value})

        return {
            'selected_numerical_states': selected_states,
            'selected_symbolic_states': symbolic_selected_states,
            'total_numerical_sum': total_value,
            'simplified_symbolic_expression': simplified_expr,
            'simplified_value': simplified_value
        }
    else:
        print(f"No suitable combination found for λ = {lambda_value} within the given tolerance.")
        return None






def simulate_LGOL_for_averages(
    order, sample_number, size, lamda, 
    t=100000, 
    start_save=50000, 
    save_interval=1, 
    save_with_simulation=False, 
    filename='LGOL_simulation_data.csv',
    box_counting_flag=False
):
    """
    Simulates the Logistic Game of Life (LGOL) for a given order, sample number, size, and lambda.
    Returns size counts, activity rates, box counting results, and averages.
    Optionally saves the data to a CSV file during simulation if save_with_simulation is True.

    Parameters:
        order (int): Order of the Cantor set.
        sample_number (int): Sample number identifier.
        size (int): Size of the grid (N x N).
        lamda (float): Lambda value for the Cantor set.
        t (int): Total number of time steps. Default is 100000.
        start_save (int): Time step to start saving data. Default is 50000.
        save_interval (int): Interval of time steps to save data. Default is 1.
        save_with_simulation (bool): If True, saves data to CSV during simulation.

    Returns:
        dict: Contains size_counts_list, activity_list, box_counts_list, averages_dict
    """
    N = M = size
    t1, t2, t3 = 1.5, 2.5, 3.5 
    
    # Initialize the grid and parameters
    L = 2**(order + 1)
    a  = np.random.randint(L, size=(M, M))
    cantor = generate_cantor_set(order, lamda)
    kernel = np.array([[1,1,1], [1,0,1], [1,1,1]])
    
    # Initialize lists to collect data
    size_counts_list = []
    activity_list = []
    box_counts_list = []
    
    # Initialize previous grid for activity calculation
    prev_grid = a.copy()
    
    if save_with_simulation:
        # Prepare CSV file
        with open(activity_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['lambda', 'time_step', 'order', 'sample_number', 'size', 'size_counts', 'state_cluster_counts', 'activity_rate', 'box_sizes', 'box_counts'])
    
    # Simulation loop
    for n in trange(t):
        s = cantor[a]
        b = convolve2d(s, kernel, mode='same', boundary='wrap')
        a_new = (L//2 + a//2)*(b >= t2)*(b <= t3) + (a)*(b >= t1)*(b < t2) + (a//2)*((b < t1) + (b > t3))
        
        if n >= start_save and n % save_interval == 0:
            # Clustering
            size_counts, state_cluster_counts = analyze_clusters(a)
            size_counts_str = ";".join(["{}:{}".format(k, v) for k, v in size_counts.items()])
            state_cluster_counts_str = ";".join(["{}:{}".format(k, v) for k, v in state_cluster_counts.items()])
            
            # Activity rate
            activity_rate = calculate_activity_rate(a_new, prev_grid)
            activity_list.append(activity_rate)
            
            if box_counting_flag:
                box_sizes, box_counts = box_counting(a)
            else:
                box_sizes, box_counts = [], []

            box_counts_list.append((box_sizes, box_counts))
            if save_with_simulation:
                with open(filename, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        lamda,
                        n,
                        order,
                        sample_number,
                        size,
                        size_counts_str,
                        state_cluster_counts_str,
                        activity_rate,
                        ";".join(map(str, box_sizes)),
                        ";".join(map(str, box_counts))
                    ])
            else:
                # Collect data for returning
                size_counts_list.append(size_counts)
        
        # Update previous grid
        prev_grid = a.copy()
        a = a_new.copy()
    
    if not save_with_simulation:
        # Compute averages
        avg_activity = np.mean(activity_list)
        if box_counting_flag:
            avg_box_counts = {
                box_size: np.mean([counts[i] for counts in box_counts_list])
                for i, box_size in enumerate(box_sizes)
            }
        else:
            avg_box_counts = []
        averages_dict = {
            'average_activity_rate': avg_activity,
            'average_box_counts': avg_box_counts,
        }
        return {
            'size_counts_list': size_counts_list,
            'activity_list': activity_list,
            'box_counts_list': box_counts_list,
            'averages': averages_dict
        }
    else:
        print(f"Simulation data saved to {filename}")
        return None