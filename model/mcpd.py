from pathlib import Path
from best_psi import best_psi
from best_threshold import best_threshold
from point_score import point_score
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Must come before importing pyplot


import matplotlib.pyplot as plt
import path
import pandas as pd
import os
from ruptures import Pelt
from scipy.signal import find_peaks
def pelt_cpd(Y, model="rbf", pen=10):
    """
    Change point detection using PELT
    """
    algo = Pelt(model=model).fit(Y)
    cp_indices = algo.predict(pen=pen)
    # Remove the last point as ruptures returns length as last cp
    return np.array(cp_indices[:-1])

def mcpd(Y, win_size=40, alpha=1):
    """
    Multiple Change Point Detection (MCPD) via model
    """
    pscore, _, _ = best_psi(Y, win_size)
    threshold, _ = best_threshold(pscore, alpha)
    indices = find_peaks(pscore, height=threshold)[0] #Non-Maximum Suppression
    interval = np.where(pscore > threshold)[0]
    #np.where(pscore > threshold)[0]
    return indices, interval, pscore, threshold

def group_cp_regions(timestamps, indices, min_gap=1):
    """Group nearby change points into (start, end) time regions"""
    if len(indices) == 0:
        return []
    indices = sorted(indices)
    regions = []
    start_idx = indices[0]
    last_time = timestamps[start_idx]
    for idx in indices[1:]:
        current_time = timestamps[idx]
        if current_time - last_time > min_gap:
            regions.append((timestamps[start_idx], last_time))
            start_idx = idx
        last_time = current_time
    regions.append((timestamps[start_idx], last_time))
    return regions

# Load data
base_path = Path(path.DATA_PATH,'b1d5f67d3ee6e58b85238a74e11cbb7a2b1881b831731ae2eb2ed1792e121638')
print("Base path:", base_path)
print(path.DATA_PATH, path.PLOTS_PATH)
gsr_pf = pd.read_csv(os.path.join(base_path, "GSR.csv"))
bvp_pf = pd.read_csv(os.path.join(base_path, "BVP.csv"))

# Convert timestamps from ms to seconds
X = gsr_pf.loc[gsr_pf['shortNTPTime'].notna(), 'shortNTPTime'].values / 1000
X2 = bvp_pf.loc[bvp_pf['shortNTPTime'].notna(), 'shortNTPTime'].values / 1000


Y = gsr_pf[['GSR_clean', 'GSR_tonic', 'GSR_phasic', 'GSR_avg', 'GSR_std']].values[gsr_pf['shortNTPTime'].notna()]
Y2 = bvp_pf[['BVP_clean', 'BVP_rate', 'BVP_avg', 'BVP_std']].values[bvp_pf['shortNTPTime'].notna()]
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(12, 9))

gsr_valid = gsr_pf.dropna(subset=['shortNTPTime'])
bvp_valid = bvp_pf.dropna(subset=['shortNTPTime'])
ax[0].plot(X, gsr_valid['GSR_clean'].values, color='steelblue')
ax[2].plot(X2, bvp_valid['BVP_clean'].values, color='steelblue')
print("Y shape:", Y.shape)
print("Y2 shape:", Y2.shape)
print("Y reshape " , Y.reshape(-1, 1).shape)
print("Y2 reshape " , Y2.reshape(-1, 1).shape)

cp_indices_max, cp_indices, pscore, threshold = mcpd(Y, win_size=50, alpha=1)
cp_indices_2_max, cp_indices_2, pscore2, threshold2 = mcpd(Y2, win_size=300, alpha=2)
print(cp_indices_max, cp_indices_2, pscore, pscore2)
print("CP Indices GSR:", cp_indices_max)
print("CP Indices BVP:", cp_indices_2_max)
print("Point Score GSR:", pscore)
print("Point Score BVP:", pscore2)
print("CP Indices Shape: ", cp_indices_max.shape)
print("Point Score Shape: ", pscore.shape)
print("CP Indices 2 Shape: ", cp_indices_2_max.shape)
print("Point Score 2 Shape: ", pscore2.shape)

'''
# Stub for GSR
cp_indices_max = np.array([50, 150, 300])
cp_indices = np.array([45, 46, 47, 145, 146, 295, 296])
pscore = np.random.rand(len(Y))  # Fake point scores
threshold = 0.6

# Stub for BVP
cp_indices_2_max = np.array([100, 400, 600])
cp_indices_2 = np.array([95, 96, 97, 395, 396, 595, 596])
pscore2 = np.random.rand(len(Y2))  # Fake point scores
threshold2 = 0.5
print(X.shape, X2.shape)
'''
# GSR plot
for _, row in gsr_pf.dropna(subset=['emotion_HRI']).iterrows():
    ax[0].axvline(x=row['shortNTPTime'] / 1000, color='black',linewidth=0.5, linestyle='-')
for idx in cp_indices_max:
    ax[0].axvline(X[idx], color='red', linestyle='-', linewidth=0.5)


ax[0].set_ylabel('GSR')
ax[0].set_xlabel('Time (seconds)')



# GSR point score
ax[1].plot(X, pscore, color='steelblue')
ax[1].axhline(y=threshold, color='black', linewidth=0.5)

# Position threshold text on right side
xlim = ax[1].get_xlim()
ax[1].text(0.99*xlim[1], threshold, 'Threshold', color='black', fontsize=8,
           verticalalignment='bottom', horizontalalignment='right')

ax[1].set_ylabel('Score')
ax[1].set_xlabel('Time (seconds)')
# BVP plot
for _, row in bvp_pf.dropna(subset=['emotion_HRI']).iterrows():
    ax[2].axvline(x=row['shortNTPTime'] / 1000, color='black', linestyle='-', linewidth=0.5)

for idx in cp_indices_2_max:
    ax[2].axvline(X2[idx], color='red', linestyle='-', linewidth=0.5)
ax[2].set_ylabel('BVP')
ax[2].set_xlabel('Time (seconds)')

# BVP point score
ax[3].plot(X2, pscore2, color='steelblue')
ax[3].axhline(y=threshold2, color='black', linewidth=0.5)

# Position threshold text on right side
xlim2 = ax[3].get_xlim()
ylim2 = ax[3].get_ylim()
ax[3].text(0.99*xlim2[1], threshold2, 'Threshold', color='black', fontsize=8,
           verticalalignment='bottom', horizontalalignment='right')

ax[3].set_ylabel('Score')

# X-axis label centered below bottom subplot
ax[3].set_xlabel('Time (seconds)')
# Remove legends if any


ax[1].set_yticks([0, 0.5, 1])
ax[3].set_yticks([0, 0.5, 1])

for a in ax:
    if a.legend_:
        a.legend_.remove()
plt.tight_layout()
plt.show()
plt.savefig( path.PLOTS_PATH / "mcpd_plot.png", dpi=300, bbox_inches='tight')