import numpy as np
from aNNEspace import aNNEspace
from scipy.spatial.distance import cdist
def point_score(Y, psi, window):
    """
    Calculate each point dissimilarity score

    Parameters:
    - Y: np array of shape (n, d), the data points
    - psi: int, the number of partitions within each partitioning
    - window: int, the size of the moving window
    """
    Y = (Y - np.min(Y)) / (1.0 * (np.max(Y) - np.min(Y)))  # Normalize Y to [0, 1]
    Y[np.isnan(Y)] = 0.5  # Replace NaNs with 0
    type = 'NormalisedKernel'

    Sdata = Y
    data = Y

    t = 200

    ndata = aNNEspace(Sdata, data, psi, t)

    # index each segmentation
    n = int(Y.shape[0])

    '''
    index = np.arange(0, n, window)
    if index[-1] != n:
        index = np.append(index, n)
    # Kernel mean embedding per segment
    # mdata : mean of each segment
    mdata = []

    for i in range(len(index) - 1):
        start, end = index[i], index[i + 1]
        segment = ndata[start:end, :]
        mdata.append(np.mean(segment, axis=0))

    mdata = np.array(mdata)
    
     # k-nn comparison with k = 1
    k = 1
    score = []

    if type == 'NormalisedKernel':
        for i in range(k, mdata.shape[0]):
            cscore = []
            for j in range(1, k + 1):
                nom = np.dot(mdata[i], mdata[i - j])
                denom = (np.dot(mdata[i], mdata[i]) ** 0.5) * (np.dot(mdata[i - j], mdata[i - j]) ** 0.5)
                cscore.append(nom / denom)
            score.append(1 - np.mean(cscore))
        score = [0] * k + score  # prepend zeros for first k segments (only 1 here)
    elif type == 'MMD':
        for i in range(k, mdata.shape[0]):
            dists = cdist(mdata[i:i + 1], mdata[i - k:i], metric='euclidean')
            score.append(np.mean(dists))
        score = [0] * k + score
    
    # Assign scores to each point within a segment for all segments
    pscore = np.zeros(int(n))
    for i in range(0, len(index) - 1):
        start, end = index[i], index[i + 1]
        pscore[start:end] = score[i]

    # Normalize scores to [0, 1]
    pscore = np.array(pscore, dtype=np.float32)
    denom = np.max(pscore) - np.min(pscore)
    if denom != 0:
        pscore = (pscore - np.min(pscore)) / denom
    else:
       pscore.fill(0)

    return pscore
    
    
    '''

    # for CPD:---------------
    mdata = []
    startL = np.arange(0, n - 2 * window)
    endL = np.arange(window - 1, n - window - 1)
    startR = np.arange(window + 1, n - window + 1)
    endR = np.arange(2 * window - 1, n)
    index = list(zip(startL, endL, startR, endR))
    index = np.array(index).flatten()
    print("index: ", str(index))
    for i in range(0, len(index) - 1, 2):
        start, end = index[i] , index[i + 1]
        segment = ndata[start:end, :]
        mdata.append(np.mean(segment, axis=0))

    mdata = np.array(mdata)

    # k-nn comparison with k = 1
    k = 1
    score = []


    for i in range(k, mdata.shape[0], 2):
        cscore = []
        for j in range(1, k + 1):
            nom = np.dot(mdata[i], mdata[i - j])
            denom = (np.dot(mdata[i], mdata[i]) ** 0.5) * (np.dot(mdata[i - j], mdata[i - j]) ** 0.5)
            cscore.append(nom / denom)
        score.append(1 - np.mean(cscore))
    score = [0] * k + score  # prepend zeros for first k segments (only 1 here)
    print("SCORE len", len(score)-1)
    pscore = np.zeros(int(n))
    print("Pscore len", pscore.shape)
    print("loop len" , len(index) - 2 * window)
    for i in range(0, len(score)):
        pscore[i + window ] = score[i]

    # Normalize scores to [0, 1]
    pscore = np.array(pscore, dtype=np.float32)
    denom = np.max(pscore) - np.min(pscore)
    if denom != 0:
        pscore = (pscore - np.min(pscore)) / denom
    else:
        pscore.fill(0)
    return pscore

    #------------------------



