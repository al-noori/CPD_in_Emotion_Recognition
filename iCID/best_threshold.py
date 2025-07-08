import numpy as np


def best_threshold(pscore, a):
    """
    Adjust the threshold by changing the value of a (alpha).

    Parameters:
    - Pscore: np.ndarray or list, point dissimilarity scores
    - a: float, alpha value to adjust threshold

    Returns:
    - best_threshold_val: float, computed threshold
    - result: np.ndarray of 0s and 1s, labels based on threshold
    """
    pscore = np.array(pscore)  # Ensure Pscore is a numpy array

    best_threshold_val = np.mean(pscore) + a * np.std(pscore)

    # Label points: 1 if Pscore > threshold else 0
    result = np.where(pscore > best_threshold_val, 1, 0)

    return best_threshold_val, result
