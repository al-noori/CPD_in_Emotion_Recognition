import numpy as np
from scipy.spatial.distance import cdist
# Nearest Neighbor Embedding (NNE) space transformation function
def aNNEspace(Sdata, data, psi, t):
    """
    Randomized approximate nearest neighbor embedding.

    Parameters:
    - Sdata: numpy array of shape (sn, d)
    - data: numpy array of shape (n, d)
    - psi: int, number of partitions within each partitioning
    - t: int, finite number of partitionings

    Returns:
    - ndata: numpy array of shape (n, t * psi), the transformed feature space
    """
    sn = Sdata.shape[0]
    n = data.shape[0]
    ndata = []

    for _ in range(t):
        # Randomly sample psi indices from Sdata without replacement
        sub_indices = np.random.choice(sn, size=psi, replace=False)
        tdata = Sdata[sub_indices, :]

        # Compute pairwise distances between sampled tdata and data
        distances = cdist(tdata, data)  # shape: (psi, n)

        # For each column (data point), find the index of the nearest tdata point
        center_idx = np.argmin(distances, axis=0)  # shape: (n,)

        # Create binary matrix z of shape (psi, n), one-hot encoded
        z = np.zeros((psi, n), dtype=np.float32)
        z[center_idx, np.arange(n)] = 1

        # Transpose to (n, psi) and append to ndata
        ndata.append(z.T)

    # Concatenate all (n, psi) blocks horizontally to form (n, t*psi)
    return np.hstack(ndata)
