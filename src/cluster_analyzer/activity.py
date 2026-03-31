import numpy as np

def calculate_activity_rate(current_state, past_state):
    """
    Calculates the activity rate between two grid states.

    Parameters:
        current_state (np.array): The current grid state.
        past_state (np.array): The previous grid state.

    Returns:
        float: The proportion of cells that have changed state.
    """
    different_state_count = np.sum(current_state != past_state)
    return different_state_count / current_state.size
