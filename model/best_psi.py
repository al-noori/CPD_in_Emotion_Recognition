import numpy as np
from joblib import Parallel, delayed
from point_score import point_score
from scipy.stats import entropy
from scipy.spatial import KDTree

# Entropy Approximation according to Pincus, 1991 with a batch size parameter

def _max_dist(x_i, x_j):
    return np.max(np.abs(x_i - x_j))

def _phi(m, r, data, batch_size=1000):
    N = len(data)
    x = np.array([data[i:i+m] for i in range(N - m + 1)])
    C = []
    for i in range(0, x.shape[0], batch_size):
        count = 0
        for j in range(0,len(x)):
            if _max_dist(x[i], x[j]) <= r:
                count += 1
        C.append(count / (N - m + 1))

    C = np.array(C)
    return np.sum(np.log(C)) / (N - m + 1)
'''

def _phi(m, r, data, batch_size=1000):
    N = len(data)
    x = np.array([data[i:i + m] for i in range(N - m + 1)])
    tree = KDTree(x)

    counts = []
    for i in range(0, len(x), batch_size):
        batch = x[i:i + batch_size]
        neighbors = tree.query_ball_point(batch, r)
        counts.extend([len(c) / (N - m + 1) for c in neighbors])
    counts = np.array(counts)

    return np.sum(np.log(counts)) / (N - m + 1)
'''
def approximate_entropy(data, m, r):
    data = np.ravel(data)  # flatten in case input shape is (N,1)
    return _phi(m, r, data) - _phi(m + 1, r, data)

def best_psi(Y, window):
    psi_list = np.array([2, 4, 8, 16, 32, 64])

    results = Parallel(n_jobs=-1)(delayed(_score_and_entropy)(Y, psi, window) for psi in psi_list)
    pscore_list, ent_list = zip(*results)
    best_idx = np.argmin(ent_list)
    return pscore_list[best_idx], ent_list[best_idx], psi_list[best_idx]


def _score_and_entropy(Y, psi, window):
    pscore = point_score(Y, psi, window)
    ent = approximate_entropy(pscore, m=2, r=0.2*np.std(pscore))
    return pscore, ent